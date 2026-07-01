# Auto-added for distilled checkpoint evaluation.
import os
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_eval_ckpt_cfg = EasyDict(__name__='Config: VA robotwin distilled eval checkpoint')
va_robotwin_eval_ckpt_cfg.update(va_robotwin_cfg)

va_robotwin_eval_ckpt_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    'EVAL_MODEL_PATH',
    '/mnt/afs/intern/manlichen/ivan/zhoujunl/models/FlashWAM-RoboTwin',
)
va_robotwin_eval_ckpt_cfg.num_inference_steps = int(os.environ.get('EVAL_VIDEO_STEPS', '1'))
va_robotwin_eval_ckpt_cfg.video_exec_step = -1
va_robotwin_eval_ckpt_cfg.action_num_inference_steps = int(os.environ.get('EVAL_ACTION_STEPS', '2'))
