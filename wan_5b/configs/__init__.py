# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_ti2v_5B import ti2v_5B

WAN_CONFIGS = {'ti2v-5B': ti2v_5B}
