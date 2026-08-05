# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
from torch import nn
import torch.distributed as dist
import torch
import math

from pipeline import SelfForcingTrainingPipeline
from utils.config import resolve_model_location, section_get
from utils.loss import get_denoising_loss
from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def build_default_denoising_step_list(sampling_steps, num_train_timesteps=1000, shift=1.0, include_zero=True):
    sigmas = torch.linspace(1.0, 0.0, int(sampling_steps) + 1, dtype=torch.float32)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = (sigmas * num_train_timesteps).to(torch.long)
    if include_zero:
        timesteps = torch.cat([timesteps, torch.zeros(1, dtype=torch.long)])
    return timesteps


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        print("args.model_kwargs.model_name", args.model_kwargs.model_name)
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.independent_first_frame = False
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.denoising_step_list = build_default_denoising_step_list(
            sampling_steps=args.sampling_steps,
            num_train_timesteps=args.num_train_timestep,
            shift=args.timestep_shift,
            include_zero=True,
        )

    def _initialize_models(self, args, device):
        self.local_attn_size = section_get(
            args,
            "inference",
            "local_attn_size",
            getattr(args, "model_kwargs", {}).get("local_attn_size", -1),
            aliases=("inference_local_attn_size",),
        )
        score_is_causal = True
        sequence_parallel_size = int(getattr(args, "sequence_parallel_size", 1))

        model_name, model_root = resolve_model_location(args.model_kwargs)
        if "5B" not in model_name:
            raise ValueError(f"Only Wan2.2-TI2V-5B is supported in this release, got {model_name}")
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Using all-causal 5B mode")

        # Generator
        generator_kwargs = dict(getattr(args, "model_kwargs", {}))
        generator_kwargs["use_ulysses_sp"] = sequence_parallel_size > 1
        self.generator = WanDiffusionWrapper(**generator_kwargs, is_causal=True)
        self.generator.model.requires_grad_(True)

        # Real Score
        real_kwargs = args.real_model_kwargs
        self.real_score = WanDiffusionWrapper(**real_kwargs, is_causal=score_is_causal)
        self.real_score.model.requires_grad_(False)

        # Fake Score
        fake_kwargs = args.fake_model_kwargs
        self.fake_score = WanDiffusionWrapper(**fake_kwargs, is_causal=score_is_causal)
        self.fake_score.model.requires_grad_(True)

        # Text Encoder & VAE
        self.text_encoder = WanTextEncoder(
            model_name=model_name,
            model_root=model_root,
        )
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(
            model_name=model_name,
            model_root=model_root,
        )
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            timestep = timestep.reshape(
                timestep.shape[0], -1, num_frame_per_block
            )
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(getattr(args, "denoising_loss_type", "flow"))()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        slice_last_frames: int = 21,
        noise=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._run_generator_backward_simulation(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            slice_last_frames=slice_last_frames,
            noise=noise,
        )

    def _run_generator_backward_simulation(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        slice_last_frames: int = 21,
        noise=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        On-policy generator via backward simulation (original path).
        """
        num_generated_frames = self.num_training_frames
        noise_shape = image_or_video_shape.copy()
        noise_shape[1] = num_generated_frames
        if noise is not None:
            noise = noise[:, :num_generated_frames]
        else:
            noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)
        
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=noise,
            slice_last_frames=slice_last_frames,
            **conditional_dict,
        )

        return (
            pred_image_or_video.to(self.dtype),
            None,
            denoised_timestep_from,
            denoised_timestep_to,
        )

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        slice_last_frames: int = 21,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, **conditional_dict, slice_last_frames=slice_last_frames
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        local_attn_size = section_get(
            self.args,
            "inference",
            "local_attn_size",
            getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1),
            aliases=("inference_local_attn_size",),
        )
        sink_size = section_get(
            self.args,
            "inference",
            "sink_size",
            getattr(self.args, "model_kwargs", {}).get("sink_size", 0),
            aliases=("inference_sink_size",),
        )
        multi_shot_sink = section_get(self.args, "inference", "multi_shot_sink", False)
        multi_shot_rope_offset = section_get(
            self.args,
            "inference",
            "multi_shot_rope_offset",
            0.0,
        )
        scene_cut_prefix = section_get(self.args, "inference", "scene_cut_prefix", "[SCENE_CUT]")
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        # do not use self.num_training_frames, because it is changed by generator_loss and critic_loss
        num_training_frames = getattr(self.args, "num_training_frames")
        if local_attn_size == -1:
            kv_cache_size = num_training_frames
        else:
            kv_cache_size = min(local_attn_size + slice_last_frames, num_training_frames)
        frame_seq_length = math.prod(self.args.image_or_video_shape[-2:]) // 4
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            same_step_across_blocks=getattr(
                self.args, "same_step_across_blocks", getattr(self, "same_step_across_blocks", False)
            ),
            last_step_only=getattr(self.args, "last_step_only", False),
            num_max_frames=kv_cache_size,
            context_noise=getattr(self.args, "context_noise", 0),
            sampling_steps=getattr(self.args, "sampling_steps", None),
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            multi_shot_sink=multi_shot_sink,
            scene_cut_prefix=scene_cut_prefix,
            multi_shot_rope_offset=multi_shot_rope_offset,
            slice_last_frames=slice_last_frames,
            num_training_frames=num_training_frames,
            model_name=getattr(self.args, "model_kwargs", {}).get("model_name", None),
            frame_seq_length=frame_seq_length,
        )
