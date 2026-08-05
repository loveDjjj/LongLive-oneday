import torch


def _overwrite_i2v_context(
    image_or_video: torch.Tensor,
    initial_latent: torch.Tensor | None,
    context_frames: int,
) -> torch.Tensor:
    """Preserve optional context frames used by shared inference pipelines."""
    if context_frames <= 0:
        return image_or_video
    output = image_or_video.clone()
    output[:, :context_frames] = initial_latent[:, :context_frames].to(
        device=output.device,
        dtype=output.dtype,
    )
    return output


def _zero_i2v_context_timestep(
    timestep: torch.Tensor,
    context_frames: int,
) -> torch.Tensor:
    if context_frames <= 0:
        return timestep
    output = timestep.clone()
    output[:, :context_frames] = 0
    return output
