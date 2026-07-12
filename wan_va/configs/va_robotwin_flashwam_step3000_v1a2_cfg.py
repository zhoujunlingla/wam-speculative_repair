# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg


va_robotwin_flashwam_step3000_v1a2_cfg = EasyDict()
va_robotwin_flashwam_step3000_v1a2_cfg.update(va_robotwin_cfg)
va_robotwin_flashwam_step3000_v1a2_cfg.__name__ = (
    "Config: RoboTwin FlashWAM official step3000 v1/a2"
)
va_robotwin_flashwam_step3000_v1a2_cfg.wan22_pretrained_model_name_or_path = (
    "/mnt/afs/intern/manlichen/ivan/zhoujunl/models/FlashWAM_eval_official_v1a2/"
    "step_3000_target_student_resume_g0124_cleanrestart"
)
va_robotwin_flashwam_step3000_v1a2_cfg.num_inference_steps = 1
va_robotwin_flashwam_step3000_v1a2_cfg.video_exec_step = -1
va_robotwin_flashwam_step3000_v1a2_cfg.action_num_inference_steps = 2

