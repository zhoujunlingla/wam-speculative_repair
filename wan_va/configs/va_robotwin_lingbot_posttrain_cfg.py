# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_lingbot_posttrain_cfg = EasyDict(__name__='Config: VA robotwin LingBot posttrain teacher')
va_robotwin_lingbot_posttrain_cfg.update(va_robotwin_cfg)

va_robotwin_lingbot_posttrain_cfg.wan22_pretrained_model_name_or_path = (
    "/mnt/afs/intern/manlichen/ivan/zhoujunl/models/lingbot-va-posttrain-robotwin"
)
va_robotwin_lingbot_posttrain_cfg.num_inference_steps = 25
va_robotwin_lingbot_posttrain_cfg.video_exec_step = -1
va_robotwin_lingbot_posttrain_cfg.action_num_inference_steps = 50
