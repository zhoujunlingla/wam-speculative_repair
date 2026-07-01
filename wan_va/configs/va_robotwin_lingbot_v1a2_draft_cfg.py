# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_lingbot_v1a2_draft_cfg = EasyDict()
va_robotwin_lingbot_v1a2_draft_cfg.update(va_robotwin_cfg)
va_robotwin_lingbot_v1a2_draft_cfg.__name__ = 'Config: VA robotwin LingBot posttrain v1/a2 draft'

va_robotwin_lingbot_v1a2_draft_cfg.wan22_pretrained_model_name_or_path = (
    "/mnt/afs/intern/manlichen/ivan/zhoujunl/models/lingbot-va-posttrain-robotwin"
)
va_robotwin_lingbot_v1a2_draft_cfg.num_inference_steps = 1
va_robotwin_lingbot_v1a2_draft_cfg.video_exec_step = -1
va_robotwin_lingbot_v1a2_draft_cfg.action_num_inference_steps = 2
