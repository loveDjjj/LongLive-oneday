# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import gc
import json
import logging

from utils.dataset import (
    DEFAULT_SCENE_CUT_PREFIX,
    PromptDataset,
    ResumableDistributedSampler,
    cycle,
    prompt_collate_fn,
)
from utils.config import section_get, wan_default_config
from utils.distributed import fsdp_wrap, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
from utils.training_state import (
    capture_rng_state,
    find_latest_training_checkpoint,
    list_training_checkpoints,
    resume_samples_per_rank,
    restore_fsdp_optimizer_state,
    restore_rng_state,
    validate_sparse_checkpoint_method,
)
from utils.device import current_device, empty_cache
from utils.inference_utils import (
    clean_fsdp_state_dict_keys,
    configure_generator_linear_only,
    is_sla_linear_parameter,
    load_generator_linear_state_dict,
    load_generator_state_dict,
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)
from torchvision.io import write_video
from tqdm.auto import tqdm

# LoRA related imports
import peft
from peft import get_peft_model_state_dict

from pipeline import (
    CausalDiffusionInferencePipeline
)
import time

class Trainer:
    
    def __init__(self, config):
        self.config = config
        self.step = 0
        self._resume_training_state = None

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.sequence_parallel_size = int(getattr(config, "sequence_parallel_size", 1))
        if self.world_size % self.sequence_parallel_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by "
                f"sequence_parallel_size ({self.sequence_parallel_size})"
            )
        self.data_parallel_size = self.world_size // self.sequence_parallel_size
        self.data_parallel_rank = global_rank // self.sequence_parallel_size
        self.sp_group = None
        self.dp_group = None
        if self.sequence_parallel_size > 1:
            from wan_5b.distributed.sp_training import (
                build_sp_dp_rank_layout,
                set_data_parallel_group,
                set_sequence_parallel_group,
                validate_sequence_parallel_training_config,
            )
            from wan_5b.distributed.sp_ulysses_inference import init_sequence_parallel

            validate_sequence_parallel_training_config(
                config, self.sequence_parallel_size, config.num_frame_per_block
            )
            sp_rank_lists, dp_rank_lists = build_sp_dp_rank_layout(
                self.world_size, self.sequence_parallel_size
            )
            sp_groups = [dist.new_group(ranks=ranks) for ranks in sp_rank_lists]
            self.sp_group = sp_groups[self.data_parallel_rank]
            set_sequence_parallel_group(self.sp_group)
            init_sequence_parallel(group=self.sp_group)

            dp_groups = [dist.new_group(ranks=ranks) for ranks in dp_rank_lists]
            self.dp_group = dp_groups[global_rank % self.sequence_parallel_size]
            set_data_parallel_group(self.dp_group)

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = current_device()
        self.is_main_process = global_rank == 0
        self._train_progress = None
        self.causal = getattr(config, "causal", getattr(config, "all_causal", True))
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        # Ranks in one SP group must sample identical noise/timesteps because
        # they jointly execute one logical sample.
        set_seed(config.seed + self.data_parallel_rank)

        if self.is_main_process and self.sequence_parallel_size > 1:
            print(
                f"[SP-DP] enabled: SP={self.sequence_parallel_size}, "
                f"DP={self.data_parallel_size}, world={self.world_size}"
            )

        if self.is_main_process and not self.disable_wandb:
            if getattr(config, "wandb_key", None):
                wandb.login(key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                id=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
                resume="allow"
            )

        self.output_path = config.output_dir
        self.metrics_path = getattr(config, "metrics_path", "")
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            if self.metrics_path:
                os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

        # Step 2: Initialize the model
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError(f"Unsupported distribution matching loss: {config.distribution_loss}")
        if not getattr(config, "adapter", None):
            raise ValueError("Maintained SLA+CAG training requires a LoRA adapter")

        # Auto resume configuration (needed for LoRA checkpoint loading)
        auto_resume = getattr(config, "auto_resume", True)  # Default to True

        # ================================= LoRA Configuration =================================
        self.is_lora_enabled = False
        self.lora_config = None
        self.generator_train_scope = str(
            getattr(config, "generator_train_scope", "lora")
        )
        if self.generator_train_scope not in {"lora", "linear_only"}:
            raise ValueError(
                "training.generator_train_scope must be lora or linear_only"
            )

        if hasattr(config, 'adapter') and config.adapter is not None:
            self.is_lora_enabled = True
            self.lora_config = config.adapter
            
            if self.is_main_process:
                print(f"LoRA enabled with config: {self.lora_config}")
                print("Loading base model and applying LoRA before FSDP wrapping...")
            
            # 1. Load base model first (config.generator_ckpt) before applying LoRA.
            generator_checkpoint_path = getattr(config, "generator_ckpt", None)
            if generator_checkpoint_path:
                if self.is_main_process:
                    print(f"Loading base model from {generator_checkpoint_path} (before applying LoRA)")
                generator_checkpoint = torch.load(generator_checkpoint_path, map_location="cpu")
                
                # Load generator (directly; no key alignment needed since LoRA not applied yet)
                if isinstance(generator_checkpoint, dict) and "generator" in generator_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {generator_checkpoint_path}")
                    load_generator_state_dict(
                        self.model.generator,
                        generator_checkpoint["generator"],
                        strict=True,
                    )
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                elif isinstance(generator_checkpoint, dict) and "model" in generator_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {generator_checkpoint_path}")
                    load_generator_state_dict(
                        self.model.generator,
                        generator_checkpoint["model"],
                        strict=True,
                    )
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                else:
                    load_generator_state_dict(
                        self.model.generator,
                        generator_checkpoint,
                        strict=True,
                    )
                    if self.is_main_process:
                        print("Loading base model as raw state_dict")
                
                # Load critic from full/base checkpoints when available.
                if isinstance(generator_checkpoint, dict) and "critic" in generator_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {generator_checkpoint_path}")
                    load_generator_state_dict(
                        self.model.fake_score,
                        generator_checkpoint["critic"],
                        strict=True,
                    )
                    if self.is_main_process:
                        print("Critic weights loaded successfully")
                # Load training step from checkpoint metadata.
                if isinstance(generator_checkpoint, dict) and "step" in generator_checkpoint:
                    self.step = generator_checkpoint["step"]
                    if self.is_main_process:
                        print(f"base_checkpoint step: {self.step}")
                else:
                    if self.is_main_process:
                        print("Warning: Step not found in checkpoint, starting from step 0.")
                del generator_checkpoint
                gc.collect()
            else:
                if self.is_main_process:
                    print("No base model checkpoint specified, skipping base weight loading for LoRA training.")

        # Apply LoRA wrapping if enabled (after all base weights are loaded, before FSDP)
        if self.is_lora_enabled:
            # 2. Apply LoRA wrapping now (after loading base model, before FSDP wrapping)
            if self.is_main_process:
                print("Applying LoRA to models...")
            if self.generator_train_scope == "lora":
                self.model.generator.model = self._configure_lora_for_model(
                    self.model.generator.model, "generator"
                )
            else:
                self._configure_generator_linear_only(self.model.generator.model)

            # Configure LoRA for fake_score if needed
            if getattr(self.lora_config, 'apply_to_critic', True):
                self.model.fake_score.model = self._configure_lora_for_model(self.model.fake_score.model, "fake_score")
                if self.is_main_process:
                    print("LoRA applied to both generator and critic")
            else:
                if self.is_main_process:
                    print("LoRA applied to generator only")

            # 3. Load LoRA weights before FSDP wrapping (if a checkpoint is available).
            # Priority: auto_resume -> legacy lora_ckpt -> initialized adapters.
            lora_checkpoint_path = None
            lora_checkpoint = None
            if auto_resume and self.output_path:
                latest_checkpoint = self.find_latest_checkpoint(self.output_path)
                if latest_checkpoint:
                    lora_checkpoint_path = latest_checkpoint
                    if self.is_main_process:
                        print(f"Auto resume: Found LoRA checkpoint at {lora_checkpoint_path}")
                else:
                    if self.is_main_process:
                        print("Auto resume: No LoRA checkpoint found in output directory")
            elif auto_resume:
                if self.is_main_process:
                    print("Auto resume enabled but no output directory was specified for LoRA")
            else:
                if self.is_main_process:
                    print("Auto resume disabled for LoRA")

            if lora_checkpoint_path is not None:
                lora_checkpoint = torch.load(lora_checkpoint_path, map_location="cpu")
            elif getattr(config, "lora_ckpt", None):
                lora_checkpoint_path = config.lora_ckpt
                lora_checkpoint = torch.load(lora_checkpoint_path, map_location="cpu")
                if self.is_main_process:
                    print(f"Using legacy lora_ckpt: {lora_checkpoint_path}")
            elif self.is_main_process:
                print("No LoRA checkpoint specified, starting LoRA training from scratch")

            # Load LoRA checkpoint (before FSDP wrapping)
            if lora_checkpoint is not None:
                if self.is_main_process:
                    print(f"Loading LoRA checkpoint from {lora_checkpoint_path} (before FSDP wrapping)")

                sparse_method = self.config.model_kwargs.sparse_config.get(
                    "method",
                    "hsa_cag"
                    if "keep_frames" in self.config.model_kwargs.sparse_config
                    else "sla_cag",
                )
                validate_sparse_checkpoint_method(lora_checkpoint, sparse_method)
                checkpoint_scope = lora_checkpoint.get(
                    "generator_train_scope", "lora"
                )
                if checkpoint_scope != self.generator_train_scope:
                    raise ValueError(
                        f"checkpoint generator train scope is {checkpoint_scope}, "
                        f"expected {self.generator_train_scope}"
                    )

                if self.generator_train_scope == "lora":
                    if "generator_lora" not in lora_checkpoint:
                        raise ValueError(
                            f"LoRA checkpoint {lora_checkpoint_path} is missing generator_lora. "
                            f"Found keys: {list(lora_checkpoint.keys())}"
                        )
                    if self.is_main_process:
                        print(
                            "Loading LoRA generator weights: "
                            f"{len(lora_checkpoint['generator_lora'])} keys in checkpoint"
                        )
                    peft.set_peft_model_state_dict(
                        self.model.generator.model, lora_checkpoint["generator_lora"]
                    )
                    del lora_checkpoint["generator_lora"]
                else:
                    if "generator_linear" not in lora_checkpoint:
                        raise ValueError(
                            f"linear-only checkpoint {lora_checkpoint_path} is missing "
                            f"generator_linear. Found keys: {list(lora_checkpoint.keys())}"
                        )
                    self._load_generator_linear_state(
                        self.model.generator.model, lora_checkpoint["generator_linear"]
                    )
                    del lora_checkpoint["generator_linear"]

                if getattr(self.lora_config, 'apply_to_critic', True):
                    if "critic_lora" not in lora_checkpoint:
                        raise ValueError(f"LoRA checkpoint {lora_checkpoint_path} is missing critic_lora.")
                    if self.is_main_process:
                        print(f"Loading LoRA critic weights: {len(lora_checkpoint['critic_lora'])} keys in checkpoint")
                    peft.set_peft_model_state_dict(self.model.fake_score.model, lora_checkpoint["critic_lora"])
                    del lora_checkpoint["critic_lora"]
                gc.collect()

                if "step" in lora_checkpoint:
                    self.step = lora_checkpoint["step"]
                    if self.is_main_process:
                        print(f"Resuming LoRA training from step {self.step}")
                self._resume_training_state = lora_checkpoint
            else:
                if self.is_main_process:
                    print("No LoRA checkpoint to load, starting from scratch")

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )
        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # Step 3: Initialize the optimizers
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        if self.is_lora_enabled and self._resume_training_state is not None:
            self._restore_lora_optimizer_state(self._resume_training_state)

        # Step 4: Initialize the dataloader
        model_name = config.model_kwargs.model_name
        num_frame_per_block = getattr(config, "num_frame_per_block", 1)
        self.fps = wan_default_config[model_name].get("fps", 16)

        latent_frames_for_dataset = list(config.image_or_video_shape)[1]
        num_training_frames = getattr(config, "num_training_frames", latent_frames_for_dataset)
        assert latent_frames_for_dataset >= num_training_frames, (
            f"image_or_video_shape[1] ({latent_frames_for_dataset}) must be >= "
            f"num_training_frames ({num_training_frames}), otherwise the dataset "
            f"will not provide enough prompts for the rollout."
        )
        total_frames = (latent_frames_for_dataset - 1) * wan_default_config[model_name]["temporal_compression_ratio"] + 1
        if dist.get_rank() == 0:
            print(f"[Dataset] latent_frames_for_dataset={latent_frames_for_dataset}, total_frames={total_frames}")

        temporal_compression_ratio = wan_default_config[model_name]["temporal_compression_ratio"]
        first_chunk_frames = 1 + (num_frame_per_block - 1) * temporal_compression_ratio
        subsequent_chunk_frames = num_frame_per_block * temporal_compression_ratio
        num_blocks = 1 + (total_frames - first_chunk_frames) // subsequent_chunk_frames
        dataset = PromptDataset(
            data_path=config.data_path,
            num_blocks=num_blocks,
        )
        collate_fn = prompt_collate_fn
        if dist.get_rank() == 0:
            print(
                "[data] prompt-only backward simulation: "
                f"path={config.data_path}, blocks={num_blocks}"
            )
        resume_samples = self._resume_samples_per_rank(config.batch_size)
        sampler = ResumableDistributedSampler(
            dataset,
            num_replicas=self.data_parallel_size,
            rank=self.data_parallel_rank,
            shuffle=True,
            drop_last=True,
            seed=int(config.seed),
            start_index=0,
        )
        resume_epoch, resume_sample_offset = divmod(
            resume_samples, max(sampler.num_samples, 1)
        )
        sampler.start_index = resume_sample_offset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler,
            num_workers=2, prefetch_factor=1, pin_memory=False,
            persistent_workers=False, collate_fn=collate_fn,
        )

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader, start_epoch=resume_epoch)

        # Step 5: Initialize the validation dataloader for visualization (fixed prompts)
        self.fixed_vis_batch = None
        self.vis_interval = section_get(config, "evaluation", "interval", getattr(config, "vis_interval", -1))
        if config.no_visualize:
            self.vis_interval = -1
        configured_vis_lengths = section_get(config, "evaluation", "num_frames", getattr(config, "vis_video_lengths", []))
        self.save_vis_latents_only = section_get(
            config,
            "evaluation",
            "save_latents_only",
            getattr(config, "return_latents", True),
            aliases=("return_latents", "save_latent_only"),
        )
        if isinstance(configured_vis_lengths, int):
            configured_vis_lengths = [configured_vis_lengths]
        if self.vis_interval > 0 and len(configured_vis_lengths) > 0:
            # Determine validation data path
            val_data_path = (
                getattr(config, "eval_data_path", None)
                or getattr(config, "val_data_path", None)
                or config.data_path
            )

            val_dataset = PromptDataset(
                data_path=val_data_path,
                num_blocks=num_blocks,
            )
            val_collate_fn = prompt_collate_fn

            if dist.get_rank() == 0:
                print("VAL DATASET SIZE %d" % len(val_dataset))

            sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.data_parallel_size,
                rank=self.data_parallel_rank,
                shuffle=False,
                drop_last=False,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=section_get(config, "evaluation", "val_batch_size", getattr(config, "val_batch_size", 1)),
                sampler=sampler,
                num_workers=0,
                collate_fn=val_collate_fn,
            )

            # Take the first batch as fixed visualization batch
            try:
                self.fixed_vis_batch = next(iter(val_dataloader))
            except StopIteration:
                self.fixed_vis_batch = None
            
            # ----------------------------------------------------------------------------------------------------------
            # Visualization settings
            # ----------------------------------------------------------------------------------------------------------
            # List of video lengths to visualize, e.g. [8, 16, 32]
            self.vis_video_lengths = configured_vis_lengths
            for _vl in self.vis_video_lengths:
                assert _vl <= latent_frames_for_dataset, (
                    f"vis_video_lengths entry {_vl} exceeds "
                    f"image_or_video_shape[1] ({latent_frames_for_dataset}), "
                    f"the dataset will not provide enough prompts for visualization."
                )

            if self.vis_interval > 0 and len(self.vis_video_lengths) > 0:
                self._setup_visualizer()
            
        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.previous_time = None

        if self.is_lora_enabled and self._resume_training_state is not None:
            self._restore_rng_state(self._resume_training_state)
            self._resume_training_state = None
        
        if self.is_main_process:
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            if self.gradient_accumulation_steps > 1:
                print(
                    "Effective batch size: "
                    f"{config.batch_size * self.gradient_accumulation_steps * self.data_parallel_size}"
                )

    def _restore_lora_optimizer_state(self, checkpoint):
        optimizer_keys = ("generator_optimizer", "critic_optimizer")
        missing = [key for key in optimizer_keys if key not in checkpoint]
        if missing:
            if self.is_main_process:
                print(
                    "Warning: LoRA checkpoint predates full-state resume and is missing "
                    f"{missing}; AdamW state will start fresh."
                )
            return

        restore_fsdp_optimizer_state(
            FSDP,
            self.model.generator,
            self.generator_optimizer,
            checkpoint["generator_optimizer"],
        )
        restore_fsdp_optimizer_state(
            FSDP,
            self.model.fake_score,
            self.critic_optimizer,
            checkpoint["critic_optimizer"],
        )
        if self.is_main_process:
            print("Restored generator and critic AdamW state from LoRA checkpoint")

    def _configure_generator_linear_only(self, transformer):
        trainable = configure_generator_linear_only(transformer)
        if self.is_main_process:
            count = sum(
                parameter.numel()
                for parameter in transformer.parameters()
                if parameter.requires_grad
            )
            print(
                f"Generator linear-only training: {len(trainable)} tensors, "
                f"{count} parameters"
            )

    @staticmethod
    def _load_generator_linear_state(transformer, state_dict):
        load_generator_linear_state_dict(transformer, state_dict)

    def _resume_samples_per_rank(self, batch_size):
        checkpoint = self._resume_training_state
        if checkpoint is None:
            return 0

        current_accumulation = int(
            getattr(self, "gradient_accumulation_steps", self.config.gradient_accumulation_steps)
        )

        samples, metadata = resume_samples_per_rank(
            checkpoint,
            step=self.step,
            current_data_parallel_size=self.data_parallel_size,
            current_sequence_parallel_size=self.sequence_parallel_size,
            current_batch_size=int(batch_size),
            current_accumulation_steps=current_accumulation,
        )
        if metadata["remainder"] and self.is_main_process:
            print(
                "Warning: saved global sample cursor is not divisible by the new training layout; "
                "the resumed data position is rounded down."
            )
        if (
            metadata["saved_world_size"] != self.world_size
            or metadata["saved_sequence_parallel_size"] != self.sequence_parallel_size
            or metadata["saved_data_parallel_size"] != self.data_parallel_size
            or metadata["saved_batch_size"] != int(batch_size)
            or metadata["saved_accumulation_steps"] != current_accumulation
        ) and self.is_main_process:
            print(
                "Warning: training layout changed from "
                f"world={metadata['saved_world_size']}, "
                f"SP={metadata['saved_sequence_parallel_size']}, "
                f"DP={metadata['saved_data_parallel_size']}, "
                f"batch={metadata['saved_batch_size']}, "
                f"accumulation={metadata['saved_accumulation_steps']} "
                f"to world={self.world_size}, SP={self.sequence_parallel_size}, "
                f"DP={self.data_parallel_size}, batch={batch_size}, "
                f"accumulation={current_accumulation}. Optimizer state is resharded, "
                "but sample-to-rank assignment cannot be bitwise identical."
            )
        return samples

    def _gather_rng_states(self):
        local_state = capture_rng_state()
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_state)
        return gathered

    def _restore_rng_state(self, checkpoint):
        rng_states = checkpoint.get("rng_states")
        if not rng_states or len(rng_states) != self.world_size:
            if self.is_main_process:
                print(
                    "Warning: per-rank RNG state is unavailable for this world size; "
                    "continuing from deterministic rank seeds."
                )
            return
        state = rng_states[dist.get_rank()]
        restore_rng_state(state)
        if self.is_main_process:
            print("Restored per-rank Python, NumPy, Torch, and accelerator RNG state")

    def find_latest_checkpoint(self, output_dir):
        """Find the latest current or legacy checkpoint."""
        return find_latest_training_checkpoint(output_dir)

    def get_all_checkpoints(self, output_dir):
        """Return ``(step, directory, name, state_path)`` sorted by step."""
        return list_training_checkpoints(output_dir)

    def cleanup_old_checkpoints(self, output_dir, max_checkpoints):
        """Remove old checkpoints if the number exceeds max_checkpoints.
        
        Only the main process performs the actual deletion to avoid race conditions
        in distributed training.
        """
        if max_checkpoints <= 0:
            return
        
        # Only main process should perform cleanup to avoid race conditions
        if not self.is_main_process:
            return
            
        checkpoints = self.get_all_checkpoints(output_dir)
        if len(checkpoints) > max_checkpoints:
            # Calculate how many to remove
            num_to_remove = len(checkpoints) - max_checkpoints
            checkpoints_to_remove = checkpoints[:num_to_remove]  # Remove oldest ones
            
            print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints, removing {num_to_remove} oldest ones (keeping {max_checkpoints})")
            
            import shutil
            removed_count = 0
            for step, checkpoint_dir_path, dir_name, _ in checkpoints_to_remove:
                try:
                    print(f"  Removing: {dir_name} (step {step})")
                    candidate_dirs = {
                        checkpoint_dir_path,
                        os.path.join(output_dir, f"checkpoint_model_{step}"),
                        os.path.join(
                            output_dir, "checkpoints", f"step_{step:07d}"
                        ),
                    }
                    for candidate_dir in candidate_dirs:
                        if os.path.isdir(candidate_dir):
                            shutil.rmtree(candidate_dir)
                    removed_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to remove checkpoint {dir_name}: {e}")
            
            print(f"Checkpoint cleanup completed: removed {removed_count}/{num_to_remove} old checkpoints")
        else:
            if len(checkpoints) > 0:
                print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints (max: {max_checkpoints}, no cleanup needed)")

    def save(self):
        print("Start gathering distributed model states...")

        if self.is_lora_enabled:
            rng_states = self._gather_rng_states()
            generator_state_key = (
                "generator_lora"
                if self.generator_train_scope == "lora"
                else "generator_linear"
            )
            if self.generator_train_scope == "lora":
                generator_train_state = self._gather_lora_state_dict(
                    self.model.generator.model
                )
            else:
                generator_train_state = self._gather_generator_linear_state()
            crit_lora_sd = self._gather_lora_state_dict(
                self.model.fake_score.model)

            with FSDP.state_dict_type(
                self.model.generator,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                generator_optim_state = FSDP.optim_state_dict(
                    self.model.generator, self.generator_optimizer
                )

            if dist.is_initialized():
                dist.barrier()

            with FSDP.state_dict_type(
                self.model.fake_score,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                critic_optim_state = FSDP.optim_state_dict(
                    self.model.fake_score, self.critic_optimizer
                )

            state_dict = {
                generator_state_key: generator_train_state,
                "critic_lora": crit_lora_sd,
                "generator_optimizer": generator_optim_state,
                "critic_optimizer": critic_optim_state,
                "step": self.step,
                "checkpoint_format_version": 4,
                "generator_train_scope": self.generator_train_scope,
                "generator_trainable_parameters": sorted(
                    name
                    for name, parameter in self.model.generator.named_parameters()
                    if parameter.requires_grad
                ),
                "sparse_method": str(
                    self.config.model_kwargs.sparse_config.get(
                        "method",
                        "hsa_cag"
                        if "keep_frames" in self.config.model_kwargs.sparse_config
                        else "sla_cag",
                    )
                ),
                "world_size": self.world_size,
                "sequence_parallel_size": self.sequence_parallel_size,
                "data_parallel_size": self.data_parallel_size,
                "batch_size": int(self.config.batch_size),
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "global_samples_consumed": (
                    self.step
                    * self.data_parallel_size
                    * int(self.config.batch_size)
                    * self.gradient_accumulation_steps
                ),
                "rng_states": rng_states,
            }

        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path, "checkpoints", f"step_{self.step:07d}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "train_state.pt")
            temporary_file = checkpoint_file + ".tmp"
            torch.save(state_dict, temporary_file)
            os.replace(temporary_file, checkpoint_file)
            print("Model saved to", checkpoint_file)
            
            # Cleanup old checkpoints if max_checkpoints is set
            max_checkpoints = getattr(self.config, "max_checkpoints", 0)
            if max_checkpoints > 0:
                self.cleanup_old_checkpoints(self.output_path, max_checkpoints)

        # Keep all ranks in sync so non-rank0 workers don't kick off the next
        # training iteration (and trigger NCCL watchdog timeouts) while rank0
        # is still writing the checkpoint to disk.
        if dist.is_initialized():
            dist.barrier()

        empty_cache()
        import gc
        gc.collect()

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        phase = "generator" if train_generator else "critic"
        self._set_train_progress(f"{phase}: text encoding")

        if self.step % 5 == 0:
            empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            text_prompts_flat = [p for sublist in text_prompts for p in sublist]
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts_flat)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        use_scene_cut_mask = (
            section_get(self.config, "inference", "multi_shot_sink", False)
            or section_get(
                self.config,
                "inference",
                "multi_shot_rope_offset",
                0.0,
            ) != 0.0
        )
        if use_scene_cut_mask:
            _prefix = getattr(self.config, "scene_cut_prefix", DEFAULT_SCENE_CUT_PREFIX)
            conditional_dict["scene_cut_mask"] = [
                p.startswith(_prefix) for p in text_prompts[0]
            ]

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            self._set_train_progress("generator: rollout and DMD loss")
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

            # Scale loss for gradient accumulation and backward
            self._set_train_progress("generator: backward")
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            scaled_generator_loss.backward()
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": torch.tensor(0.0, device=self.device)})

            return generator_log_dict
        else:
            generator_log_dict = {}

        self._set_train_progress("critic: rollout and loss")
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
        )

        # Scale loss for gradient accumulation and backward
        self._set_train_progress("critic: backward")
        scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
        scaled_critic_loss.backward()
        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": torch.tensor(0.0, device=self.device)})

        return critic_log_dict

    def _set_train_progress(self, stage):
        if self.is_main_process and self._train_progress is not None:
            self._train_progress.set_postfix_str(stage, refresh=True)

    def _write_metrics(self, values):
        if not self.metrics_path:
            return
        record = {"step": self.step, **values}
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def generate_video(self, pipeline, num_frames, prompts, latents_only=False):
        batch_size = len(prompts)
        sampled_noise = torch.randn(
            [batch_size, num_frames, *self.config.image_or_video_shape[2:]],
            device=self.device,
            dtype=self.dtype,
        )
        with torch.no_grad():
            result = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=latents_only,
            )
            if latents_only:
                return result
            video = result
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        if hasattr(pipeline, 'vae') and hasattr(pipeline.vae, 'model') and hasattr(pipeline.vae.model, 'clear_cache'):
            pipeline.vae.model.clear_cache()
        return current_video
    
    def train(self):
        start_step = self.step
        show_progress = os.environ.get("LLV2_TRAIN_PROGRESS", "1") != "0"
        self._train_progress = tqdm(
            total=int(self.config.max_iters),
            initial=min(self.step, int(self.config.max_iters)),
            desc="[train]",
            unit="step",
            dynamic_ncols=True,
            disable=not (self.is_main_process and show_progress),
        )
        try:
            while self.step < self.config.max_iters:
                # Check if we should train generator on this optimization step
                TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

                if TRAIN_GENERATOR:
                    self.generator_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                # Whole-cycle gradient accumulation loop
                accumulated_generator_logs = []
                accumulated_critic_logs = []

                for accumulation_step in range(self.gradient_accumulation_steps):
                    self._set_train_progress(
                        f"step {self.step}: accumulation "
                        f"{accumulation_step + 1}/{self.gradient_accumulation_steps}"
                    )
                    batch = next(self.dataloader)

                    # Train generator (if needed)
                    if TRAIN_GENERATOR:
                        extra_gen = self.fwdbwd_one_step(batch, True)
                        accumulated_generator_logs.append(extra_gen)

                    # Train critic
                    extra_crit = self.fwdbwd_one_step(batch, False)
                    accumulated_critic_logs.append(extra_crit)

                # Compute grad norm and update parameters
                self._set_train_progress(f"step {self.step}: optimizer")
                if TRAIN_GENERATOR:
                    generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                    generator_log_dict = merge_dict_list(accumulated_generator_logs)
                    generator_log_dict["generator_grad_norm"] = generator_grad_norm

                    self.generator_optimizer.step()
                else:
                    generator_log_dict = {}

                critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                critic_log_dict = merge_dict_list(accumulated_critic_logs)
                critic_log_dict["critic_grad_norm"] = critic_grad_norm

                self.critic_optimizer.step()

                # Increment the step since we finished gradient update
                self.step += 1
                if self.is_main_process:
                    self._train_progress.update(1)

                # Save the model
                if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                    empty_cache()
                    self.save()
                    empty_cache()

                # Logging
                if self.is_main_process:
                    wandb_loss_dict = {}
                    if TRAIN_GENERATOR and generator_log_dict:
                        wandb_loss_dict.update(
                            {
                                "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                                "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                                "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                            }
                        )


                    wandb_loss_dict.update(
                        {
                            "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                            "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                        }
                    )
                    if not self.disable_wandb:
                        wandb.log(wandb_loss_dict, step=self.step)

                if self.step % self.config.gc_interval == 0:
                    if dist.get_rank() == 0:
                        logging.info("DistGarbageCollector: Running GC.")
                    gc.collect()
                    empty_cache()

                if self.is_main_process:
                    current_time = time.time()
                    iteration_time = 0 if self.previous_time is None else current_time - self.previous_time
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": iteration_time}, step=self.step)
                    self.previous_time = current_time
                    self._write_metrics(
                        {**wandb_loss_dict, "iteration_seconds": iteration_time}
                    )
                    # Log training progress
                    if TRAIN_GENERATOR and generator_log_dict:
                        self._train_progress.write(f"step {self.step}, per iteration time {iteration_time}, generator_loss {generator_log_dict['generator_loss'].mean().item()}, generator_grad_norm {generator_log_dict['generator_grad_norm'].mean().item()}, dmdtrain_gradient_norm {generator_log_dict['dmdtrain_gradient_norm'].mean().item()}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")
                    else:
                        self._train_progress.write(f"step {self.step}, per iteration time {iteration_time}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")

                # ---------------------------------------- Visualization ---------------------------------------------------

                if self.vis_interval > 0 and (self.step % self.vis_interval == 0):
                    self._visualize()
                

        except Exception as e:
            print(f"[ERROR] [Rank {dist.get_rank()}] Training crashed at step {self.step} with exception: {e}")
            print(f"[ERROR] [Rank {dist.get_rank()}] Exception traceback:", flush=True)
            import traceback
            traceback.print_exc()
            raise
        finally:
            if self._train_progress is not None:
                self._train_progress.close()
                self._train_progress = None

    def _configure_lora_for_model(self, transformer, model_name):
        """Configure LoRA for a WanDiffusionWrapper model"""
        # Find all Linear modules in WanAttentionBlock modules
        target_linear_modules = set()
        
        # Define the specific modules we want to apply LoRA to
        all_causal = getattr(self.config, 'all_causal', False)
        generator_is_causal = getattr(self.config, 'generator_is_causal', True)
        if model_name == 'generator':
            adapter_target_modules = (
                ['CausalWanAttentionBlock', 'UlyssesCausalWanAttentionBlock']
                if generator_is_causal else ['WanAttentionBlock']
            )
        elif model_name == 'fake_score':
            adapter_target_modules = ['CausalWanAttentionBlock'] if all_causal else ['WanAttentionBlock']
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        for name, module in transformer.named_modules():
            if module.__class__.__name__ in adapter_target_modules:
                for full_submodule_name, submodule in module.named_modules(prefix=name):
                    sparse_config = self.config.model_kwargs.sparse_config
                    sparse_method = sparse_config.get(
                        "method",
                        "hsa_cag" if "keep_frames" in sparse_config else "sla_cag",
                    )
                    if (
                        isinstance(submodule, torch.nn.Linear)
                        and not (
                            full_submodule_name.endswith(".sla_linear")
                            and (model_name == "fake_score" or sparse_method != "sla_cag")
                        )
                    ):
                        target_linear_modules.add(full_submodule_name)
        
        target_linear_modules = list(target_linear_modules)
        
        if self.is_main_process:
            print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
            if getattr(self.lora_config, 'verbose', False):
                for module_name in sorted(target_linear_modules):
                    print(f"  - {module_name}")
        
        # Create LoRA config
        adapter_type = self.lora_config.get('type', 'lora')
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=self.lora_config.get('rank', 16),
                lora_alpha=self.lora_config.get('alpha', None) or self.lora_config.get('rank', 16),
                lora_dropout=self.lora_config.get('dropout', 0.0),
                target_modules=target_linear_modules,
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        
        # Apply LoRA to the transformer
        lora_model = peft.get_peft_model(transformer, peft_config)

        if self.is_main_process:
            print('peft_config', peft_config)
            lora_model.print_trainable_parameters()

        return lora_model


    def _gather_lora_state_dict(self, lora_model):
        "On rank-0, gather FULL_STATE_DICT, then filter only LoRA weights"
        with FSDP.state_dict_type(
            lora_model,                       # lora_model contains nested FSDP submodules
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        ):
            full = lora_model.state_dict()
        return get_peft_model_state_dict(lora_model, state_dict=full)

    def _gather_generator_linear_state(self):
        with FSDP.state_dict_type(
            self.model.generator,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            full = self.model.generator.state_dict()
        full = clean_fsdp_state_dict_keys(full)
        return {
            name.removeprefix("model."): value
            for name, value in full.items()
            if is_sla_linear_parameter(name)
        }
    
    # --------------------------------------------------------------------------------------------------------------
    # Visualization helpers
    # --------------------------------------------------------------------------------------------------------------

    def _setup_visualizer(self):
        """Initialize the maintained causal validation pipeline."""
        from copy import deepcopy

        vis_config = deepcopy(self.config)
        if "guidance_scale" not in getattr(vis_config, "inference", {}):
            vis_config.guidance_scale = 1.0
        self.vis_pipeline = CausalDiffusionInferencePipeline(
            args=vis_config,
            device=self.device,
            generator=self.model.generator,
            text_encoder=self.model.text_encoder,
            vae=self.model.vae,
        )

        # Visualization artifacts belong to the run directory, not logs/.
        self.vis_output_dir = os.path.join(self.output_path, "vis")
        os.makedirs(self.vis_output_dir, exist_ok=True)
        if section_get(self.config, "evaluation", "use_ema", getattr(self.config, "vis_ema", False)):
            raise NotImplementedError("Visualization with EMA is not implemented")

    def _visualize(self):
        """Generate validation samples to monitor training progress."""
        if self.vis_interval <= 0 or not hasattr(self, "vis_pipeline"):
            return False

        # FSDP forward includes communication, so every rank must enter
        # visualization together; running rank 0 alone would hang.

        if not getattr(self, "fixed_vis_batch", None):
            print("[Warning] No fixed validation batch available for visualization.")
            return False

        step_vis_dir = os.path.join(self.vis_output_dir, f"step_{self.step:07d}")
        os.makedirs(step_vis_dir, exist_ok=True)
        batch = self.fixed_vis_batch
        prompts = batch["prompts"]

        mode_info = ""
        if self.is_lora_enabled:
            mode_info = "_lora"
            if self.is_main_process:
                print(f"Generating latents in LoRA mode (step {self.step})")

        for vid_len in self.vis_video_lengths:
            print(f"Generating validation samples of length {vid_len}")
            samples = self.generate_video(
                self.vis_pipeline,
                vid_len,
                prompts,
                latents_only=self.save_vis_latents_only,
            )

            for idx in range(samples.shape[0]):
                if self.save_vis_latents_only:
                    sample_name = f"latents_step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.pt"
                    out_path = os.path.join(step_vis_dir, sample_name)
                    torch.save(samples[idx].cpu(), out_path)
                else:
                    sample_name = f"video_step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.mp4"
                    out_path = os.path.join(step_vis_dir, sample_name)
                    write_video(out_path, torch.as_tensor(samples[idx]).to(torch.uint8), fps=24)

            del samples
            empty_cache()

        # Save prompts for reference
        prompt_path = os.path.join(
            step_vis_dir,
            f"prompts_rank_{dist.get_rank()}.txt",
        )
        with open(prompt_path, "w") as f:
            for i, p in enumerate(prompts):
                f.write(f"[sample {i}] {p}\n")

        # Release KV / cross-attention caches allocated during inference to prevent OOM
        # when training resumes. These caches can consume ~20+ GB of GPU memory.
        if hasattr(self.vis_pipeline, 'clear_cache'):
            self.vis_pipeline.clear_cache()

        empty_cache()
        import gc
        gc.collect()

        # Synchronize all ranks so that a crashed rank is detected immediately
        # rather than causing a 10-minute NCCL timeout on the next training collective.
        dist.barrier()

        return True
