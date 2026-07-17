# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_robotwin_flashwam_step2000_v1a2_cfg import va_robotwin_flashwam_step2000_v1a2_cfg
from .va_robotwin_flashwam_step3000_v1a2_cfg import va_robotwin_flashwam_step3000_v1a2_cfg
from .va_robotwin_lingbot_v2a4_cfg import va_robotwin_lingbot_v2a4_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'robotwin_flashwam_step2000_v1a2': va_robotwin_flashwam_step2000_v1a2_cfg,
    'robotwin_flashwam_step3000_v1a2': va_robotwin_flashwam_step3000_v1a2_cfg,
    'robotwin_lingbot_v2a4': va_robotwin_lingbot_v2a4_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
    'libero': va_libero_cfg,
    'libero_train': va_libero_train_cfg,
    'libero_i2av': va_libero_i2va_cfg,
}
