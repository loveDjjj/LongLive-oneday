# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

__all__ = ["WanTI2V"]


def __getattr__(name):
    if name == "WanTI2V":
        from .textimage2video import WanTI2V

        return WanTI2V
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
