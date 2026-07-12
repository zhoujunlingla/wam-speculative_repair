# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg


va_robotwin_lingbot_v2a4_cfg = EasyDict()
va_robotwin_lingbot_v2a4_cfg.update(va_robotwin_cfg)
va_robotwin_lingbot_v2a4_cfg.__name__ = "Config: RoboTwin LingBot posttrain v2/a4"
va_robotwin_lingbot_v2a4_cfg.wan22_pretrained_model_name_or_path = (
    "/mnt/afs/intern/manlichen/ivan/zhoujunl/models/lingbot-va-posttrain-robotwin"
)
va_robotwin_lingbot_v2a4_cfg.num_inference_steps = 2
va_robotwin_lingbot_v2a4_cfg.video_exec_step = -1
va_robotwin_lingbot_v2a4_cfg.action_num_inference_steps = 4

