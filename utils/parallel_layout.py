"""Pure helpers for validating inference SP/DP layouts."""

from __future__ import annotations


def validate_sp_dp_layout(
    *, world_size: int, sp_size: int, dp_size: int, num_heads: int
) -> tuple[int, int]:
    values = {
        "world_size": world_size,
        "sp_size": sp_size,
        "dp_size": dp_size,
        "num_heads": num_heads,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if sp_size * dp_size != world_size:
        raise ValueError(
            f"parallel layout mismatch: sp_size={sp_size}, dp_size={dp_size}, "
            f"world_size={world_size}"
        )
    if num_heads % sp_size != 0:
        raise ValueError(f"sp_size={sp_size} must divide num_heads={num_heads}")
    return sp_size, dp_size
