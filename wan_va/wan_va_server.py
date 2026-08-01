# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)
from specverify import (
    action_verify_frame_start,
    make_verify_scheduler,
    quantize_prefix_to_frame_boundary,
    sample_verify_noise_like,
    scheduler_add_noise_batched,
    scheduler_step_to_final_batched,
    scheduler_step_to_timestep_batched,
)


def action_dynamics_step_score(
    action: np.ndarray,
    *,
    conditioned_frame_count: int,
    action_per_frame: int,
    delta_ref: float = 0.25,
    jerk_ref: float = 0.18,
    phase_weight: float = 0.25,
    delta_weight: float = 0.10,
    jerk_weight: float = 0.10,
) -> np.ndarray:
    """Per-step dynamics penalty for verifier gating."""

    if not isinstance(action, np.ndarray) or action.ndim != 3:
        return np.zeros((0,), dtype=np.float32)
    seq = action.transpose(1, 2, 0).reshape(-1, action.shape[0])
    score = np.zeros((seq.shape[0],), dtype=np.float32)
    cont_idx = [i for i in range(seq.shape[1]) if i not in (7, 15)]
    cont = seq[:, cont_idx] if cont_idx else seq
    if seq.shape[0] >= 2:
        delta = np.linalg.norm(np.diff(cont, axis=0), axis=1)
        score[1:] += np.clip(delta / max(delta_ref, 1e-6), 0.0, 1.0) * float(delta_weight)
    if seq.shape[0] >= 3:
        jerk = np.linalg.norm(np.diff(cont, n=2, axis=0), axis=1)
        score[2:] += np.clip(jerk / max(jerk_ref, 1e-6), 0.0, 1.0) * float(jerk_weight)
    for channel in (7, 15):
        if seq.shape[1] > channel and seq.shape[0] >= 2:
            switch = (seq[1:, channel] > 0.5) != (seq[:-1, channel] > 0.5)
            score[1:] += switch.astype(np.float32) * float(phase_weight)
    start = max(0, min(int(conditioned_frame_count) * int(action_per_frame), seq.shape[0]))
    return score[start:].astype(np.float32)



def _is_wanvae_temporal_cache_error(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "Calculated padded input size per channel" in msg
        and "Kernel size can't be greater than actual input size" in msg
    )


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.verify_scheduler = make_verify_scheduler(
            self.job_config.action_snr_shift)
        self.video_verify_scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                                         sigma_min=0.0,
                                                         extra_one_step=True)
        self.video_verify_scheduler.set_timesteps(1000, training=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        self.transformer = load_transformer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode="torch"
        )
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def forward_action_only_verify(self,
                                   noisy_actions,
                                   action_timesteps,
                                   frame_st_id=0,
                                   cache_name=None):
        """Run each verifier timestep through the teacher's full CFG batch."""
        if self.prompt_embeds is None:
            raise RuntimeError("forward_action_only_verify requires a prompt cache")
        if noisy_actions.ndim != 5:
            raise ValueError("noisy_actions must have shape [K, C, F, N, 1]")

        cache_name = cache_name or self.cache_name
        noisy_actions = noisy_actions.to(device=self.device, dtype=self.dtype).clone()
        batch_size = noisy_actions.shape[0]
        frame_chunk_size = noisy_actions.shape[2]

        timesteps = torch.as_tensor(action_timesteps,
                                    dtype=torch.float32,
                                    device=self.device).flatten()
        if timesteps.numel() == 1:
            timesteps = timesteps.repeat(batch_size)
        if timesteps.numel() != batch_size:
            raise ValueError("action_timesteps must be scalar or length K")

        conditioned_frame_count = action_verify_frame_start(frame_st_id)
        if conditioned_frame_count:
            noisy_actions[:, :, :conditioned_frame_count] = 0
        noisy_actions[:, ~self.action_mask] *= 0

        outputs = []
        for row, timestep in enumerate(timesteps):
            action_cond = torch.zeros(
                [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
                device=self.device,
                dtype=self.dtype,
            ) if conditioned_frame_count else None
            input_dict = self._prepare_latent_input(
                None,
                noisy_actions[row:row + 1],
                timestep,
                timestep,
                None,
                action_cond,
                frame_st_id=frame_st_id,
            )
            action_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict['action_res_lst']),
                update_cache=0,
                cache_name=cache_name,
                action_mode=True,
            )
            action_noise_pred = rearrange(
                action_noise_pred,
                'b (f n) c -> b c f n 1',
                f=frame_chunk_size,
            )
            if self.job_config.action_guidance_scale > 1:
                action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (
                    action_noise_pred[:1] - action_noise_pred[1:]
                )
            else:
                action_noise_pred = action_noise_pred[:1]
            outputs.append(action_noise_pred)
        return torch.cat(outputs, dim=0)

    def forward_video_only_verify(self,
                                  noisy_latents,
                                  video_timesteps,
                                  frame_st_id=0,
                                  cache_name=None):
        """Run the video branch for one verifier timestep at a time."""
        if self.prompt_embeds is None:
            raise RuntimeError("forward_video_only_verify requires a prompt cache")
        if noisy_latents.ndim != 5:
            raise ValueError("noisy_latents must have shape [K, C, F, H, W]")

        cache_name = cache_name or self.cache_name
        timesteps = torch.as_tensor(video_timesteps,
                                    dtype=torch.float32,
                                    device=self.device).flatten()
        if timesteps.numel() != noisy_latents.shape[0]:
            raise ValueError("video_timesteps must contain one value per verifier batch")

        outputs = []
        frame_chunk_size = noisy_latents.shape[2]
        for row, timestep in enumerate(timesteps):
            latent_cond = noisy_latents[row:row + 1, :, 0:1].clone() if frame_st_id == 0 else None
            input_dict = self._prepare_latent_input(
                noisy_latents[row:row + 1].to(device=self.device, dtype=self.dtype),
                None,
                timestep,
                timestep,
                latent_cond,
                None,
                frame_st_id=frame_st_id,
            )
            video_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                update_cache=0,
                cache_name=cache_name,
                action_mode=False,
            )
            video_noise_pred = data_seq_to_patch(
                self.job_config.patch_size,
                video_noise_pred,
                frame_chunk_size,
                self.latent_height,
                self.latent_width,
                batch_size=2 if self.use_cfg else 1,
            )
            if self.job_config.guidance_scale > 1:
                video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
            else:
                video_noise_pred = video_noise_pred[:1]
            outputs.append(video_noise_pred)
        return torch.cat(outputs, dim=0)

    def verify_world_latent_chunk(self,
                                  draft_latents,
                                  frame_st_id=0,
                                  tau_timesteps=(150.0, 300.0),
                                  threshold=0.35,
                                  cache_name=None):
        """Verify a draft video/world latent against the teacher video flow."""
        if draft_latents.ndim != 5 or draft_latents.shape[0] != 1:
            raise ValueError("draft_latents must have shape [1, C, F, H, W]")

        tau_timesteps = torch.as_tensor(tau_timesteps,
                                        dtype=torch.float32,
                                        device=self.device).flatten()
        if tau_timesteps.numel() < 1:
            raise ValueError("tau_timesteps must be non-empty")

        draft = draft_latents.to(device=self.device, dtype=self.dtype).clone()
        batch_size = tau_timesteps.numel()
        draft_batch = draft.repeat(batch_size, 1, 1, 1, 1)
        noise = torch.randn_like(draft).repeat(batch_size, 1, 1, 1, 1)

        conditioned_frame_count = 1 if frame_st_id == 0 else 0
        if conditioned_frame_count:
            noise[:, :, :conditioned_frame_count] = 0

        z_tau = scheduler_add_noise_batched(
            scheduler=self.video_verify_scheduler,
            clean=draft_batch,
            noise=noise,
            timesteps=tau_timesteps,
        )
        if conditioned_frame_count:
            z_tau[:, :, :conditioned_frame_count] = draft_batch[:, :, :conditioned_frame_count]

        velocity = self.forward_video_only_verify(z_tau,
                                                  tau_timesteps,
                                                  frame_st_id=frame_st_id,
                                                  cache_name=cache_name)
        recon = scheduler_step_to_final_batched(
            scheduler=self.video_verify_scheduler,
            model_output=velocity,
            timesteps=tau_timesteps,
            sample=z_tau,
        )

        dist = (recon - draft_batch).abs().mean(dim=1)
        valid_dist = dist[:, conditioned_frame_count:]
        if valid_dist.numel() == 0:
            valid_dist = dist
        # A single hot latent patch can be noisy in one-step video drafts.
        # Use a top-percentile score rather than raw max for the first WAM
        # verifier; max/p95 are both still logged by the caller.
        world_score = torch.quantile(valid_dist.float().flatten(), 0.95)
        pass_world = bool(world_score.item() <= threshold)
        return {
            "world_pass": pass_world,
            "world_score": float(world_score.item()),
            "world_score_type": "p95",
            "world_distances": valid_dist.detach().float().cpu(),
            "world_tau_timesteps": tau_timesteps.detach().float().cpu(),
            "world_threshold": float(threshold),
            "world_conditioned_frame_count": conditioned_frame_count,
        }

    def verify_action_chunk(self,
                            draft_actions,
                            frame_st_id=0,
                            tau_timesteps=(150.0, 300.0),
                            threshold=0.15,
                            cache_name=None,
                            return_repair=False,
                            repair_lambda=0.75,
                            verify_plus=False,
                            alpha_cross_tau=0.30,
                            shortcut_verify=False,
                            alpha_shortcut=0.30,
                            dynamics_gate=False,
                            dynamics_delta_ref=0.25,
                            dynamics_jerk_ref=0.18,
                            dynamics_phase_weight=0.25,
                            return_step_mask=False,
                            verify_seed=None):
        """Verify a normalized draft action chunk against the teacher flow."""
        if draft_actions.ndim != 5 or draft_actions.shape[0] != 1:
            raise ValueError("draft_actions must have shape [1, C, F, N, 1]")

        tau_timesteps = torch.as_tensor(tau_timesteps,
                                        dtype=torch.float32,
                                        device=self.device).flatten()
        if tau_timesteps.numel() < 1:
            raise ValueError("tau_timesteps must be non-empty")

        draft = draft_actions.to(device=self.device, dtype=self.dtype).clone()
        batch_size = tau_timesteps.numel()
        draft_batch = draft.repeat(batch_size, 1, 1, 1, 1)
        noise = sample_verify_noise_like(draft, verify_seed).repeat(batch_size, 1, 1, 1, 1)

        conditioned_frame_count = action_verify_frame_start(frame_st_id)
        if conditioned_frame_count:
            draft_batch[:, :, :conditioned_frame_count] = 0
            noise[:, :, :conditioned_frame_count] = 0

        z_tau = scheduler_add_noise_batched(
            scheduler=self.verify_scheduler,
            clean=draft_batch,
            noise=noise,
            timesteps=tau_timesteps,
        )
        if conditioned_frame_count:
            z_tau[:, :, :conditioned_frame_count] = 0

        velocity = self.forward_action_only_verify(z_tau,
                                                   tau_timesteps,
                                                   frame_st_id=frame_st_id,
                                                   cache_name=cache_name)
        recon = scheduler_step_to_final_batched(
            scheduler=self.verify_scheduler,
            model_output=velocity,
            timesteps=tau_timesteps,
            sample=z_tau,
        )

        dist = (recon - draft_batch).abs()[:, self.action_mask, :, :, 0]
        dist = dist.mean(dim=1)
        valid_dist = dist[:, conditioned_frame_count:, :].reshape(batch_size, -1)
        endpoint_dist_max = valid_dist.max(dim=0).values
        endpoint_dist_mean = valid_dist.mean(dim=0)

        if verify_plus and batch_size > 1:
            endpoint_std = recon[:, self.action_mask, :, :, 0].float().std(dim=0, unbiased=False)
            endpoint_std = endpoint_std.mean(dim=0)
            cross_tau_score = endpoint_std[conditioned_frame_count:, :].reshape(-1)
        else:
            cross_tau_score = torch.zeros_like(endpoint_dist_max)

        if verify_plus and shortcut_verify:
            tau_mid = torch.clamp(tau_timesteps * 0.5, min=0.0)
            z_mid = scheduler_step_to_timestep_batched(
                scheduler=self.verify_scheduler,
                model_output=velocity,
                from_timesteps=tau_timesteps,
                to_timesteps=tau_mid,
                sample=z_tau,
            )
            if conditioned_frame_count:
                z_mid[:, :, :conditioned_frame_count] = 0
            velocity_mid = self.forward_action_only_verify(
                z_mid,
                tau_mid,
                frame_st_id=frame_st_id,
                cache_name=cache_name,
            )
            recon_two_step = scheduler_step_to_final_batched(
                scheduler=self.verify_scheduler,
                model_output=velocity_mid,
                timesteps=tau_mid,
                sample=z_mid,
            )
            shortcut_dist = (recon_two_step - recon).abs()[:, self.action_mask, :, :, 0]
            shortcut_dist = shortcut_dist.mean(dim=1)
            valid_shortcut_dist = shortcut_dist[:, conditioned_frame_count:, :].reshape(batch_size, -1)
            shortcut_score = valid_shortcut_dist.max(dim=0).values
        else:
            shortcut_score = torch.zeros_like(endpoint_dist_max)

        if verify_plus and dynamics_gate:
            dynamics_np = action_dynamics_step_score(
                self.postprocess_action(draft.detach()).astype(np.float32),
                conditioned_frame_count=conditioned_frame_count,
                action_per_frame=self.action_per_frame,
                delta_ref=dynamics_delta_ref,
                jerk_ref=dynamics_jerk_ref,
                phase_weight=dynamics_phase_weight,
            )
            dynamics_score = torch.as_tensor(dynamics_np, dtype=endpoint_dist_max.dtype, device=endpoint_dist_max.device)
            if dynamics_score.numel() != endpoint_dist_max.numel():
                dynamics_score = torch.zeros_like(endpoint_dist_max)
        else:
            dynamics_score = torch.zeros_like(endpoint_dist_max)

        if verify_plus:
            step_score = (
                endpoint_dist_max
                + float(alpha_cross_tau) * cross_tau_score
                + float(alpha_shortcut) * shortcut_score
                + float(threshold) * dynamics_score
            )
        else:
            step_score = endpoint_dist_max

        pass_by_step = step_score <= float(threshold)
        raw_valid_prefix = 0
        for ok in pass_by_step.detach().cpu().tolist():
            if not ok:
                break
            raw_valid_prefix += 1

        full_prefix = conditioned_frame_count * self.action_per_frame + raw_valid_prefix
        accepted_prefix = quantize_prefix_to_frame_boundary(
            full_prefix,
            action_per_frame=self.action_per_frame,
            frame_chunk_size=self.job_config.frame_chunk_size,
        )
        result = {
            "accepted_prefix": accepted_prefix,
            "raw_valid_prefix": raw_valid_prefix,
            "conditioned_frame_count": conditioned_frame_count,
            "distances": dist.detach().float().cpu(),
            "tau_timesteps": tau_timesteps.detach().float().cpu(),
            "threshold": float(threshold),
            "verify_plus": bool(verify_plus),
            "shortcut_verify": bool(shortcut_verify),
            "pass_by_step": [bool(x) for x in pass_by_step.detach().cpu().tolist()],
            "step_score": step_score.detach().float().cpu().numpy().tolist(),
            "endpoint_dist_max": endpoint_dist_max.detach().float().cpu().numpy().tolist(),
            "endpoint_dist_mean": endpoint_dist_mean.detach().float().cpu().numpy().tolist(),
            "cross_tau_score": cross_tau_score.detach().float().cpu().numpy().tolist(),
            "shortcut_score": shortcut_score.detach().float().cpu().numpy().tolist(),
            "dynamics_score": dynamics_score.detach().float().cpu().numpy().tolist(),
            "alpha_cross_tau": float(alpha_cross_tau),
            "alpha_shortcut": float(alpha_shortcut),
            "verify_seed": int(verify_seed) if verify_seed is not None else None,
        }
        if return_repair:
            repair_lambda = max(0.0, min(1.0, float(repair_lambda)))
            teacher_endpoint = recon.mean(dim=0, keepdim=True)
            repair_latent = draft + repair_lambda * (teacher_endpoint - draft)
            if conditioned_frame_count:
                repair_latent[:, :, :conditioned_frame_count] = draft[:, :, :conditioned_frame_count]
            repair_delta = (repair_latent - draft).detach().float().abs()
            result.update({
                "repair_action_latent": repair_latent.detach().float().cpu().numpy(),
                "repair_action": self.postprocess_action(repair_latent.detach()).astype(np.float32),
                "repair_lambda": repair_lambda,
                "repair_delta_mean": float(repair_delta.mean().item()),
                "repair_delta_max": float(repair_delta.max().item()),
            })
        return result

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _reset(self, prompt=None):
        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _infer(self, obs, frame_st_id=0, teacher_self_verify=None):
        frame_chunk_size = self.job_config.frame_chunk_size
        teacher_self_verify_result = None
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                if last_step and teacher_self_verify is not None:
                    teacher_self_verify_result = self.verify_action_chunk(
                        actions.clone(),
                        frame_st_id=frame_st_id,
                        tau_timesteps=teacher_self_verify.get(
                            "tau_timesteps", (150.0, 300.0)),
                        threshold=teacher_self_verify.get("threshold", 0.15),
                    )
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0
        self.last_action_latent = actions.detach().float().cpu().numpy()
        self.last_video_latent = latents.detach().float().cpu().numpy()
        self.last_teacher_self_verify = teacher_self_verify_result

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)
        verify_action = obs.get('verify_action', False)
        verify_world_latent = obs.get('verify_world_latent', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif verify_action:
            logger.info(f"################# Verify Action Chunk #################")
            draft_action = obs.get("action_latent")
            if draft_action is None:
                raise ValueError("verify_action requires action_latent")
            if not isinstance(draft_action, torch.Tensor):
                draft_action = torch.as_tensor(draft_action)
            if draft_action.ndim == 4:
                draft_action = draft_action.unsqueeze(0)
            draft_action = draft_action.to(device=self.device, dtype=self.dtype)
            frame_st_id = int(obs.get("frame_st_id", self.frame_st_id))
            verify_result = self.verify_action_chunk(
                draft_action,
                frame_st_id=frame_st_id,
                tau_timesteps=obs.get("tau_timesteps", (150.0, 300.0)),
                threshold=float(obs.get("threshold", 0.15)),
                return_repair=bool(obs.get("return_repair", False)),
                repair_lambda=float(obs.get("repair_lambda", 0.75)),
                verify_plus=bool(obs.get("verify_plus", False)),
                alpha_cross_tau=float(obs.get("alpha_cross_tau", 0.30)),
                shortcut_verify=bool(obs.get("shortcut_verify", False)),
                alpha_shortcut=float(obs.get("alpha_shortcut", 0.30)),
                dynamics_gate=bool(obs.get("dynamics_gate", False)),
                dynamics_delta_ref=float(obs.get("dynamics_delta_ref", 0.25)),
                dynamics_jerk_ref=float(obs.get("dynamics_jerk_ref", 0.18)),
                dynamics_phase_weight=float(obs.get("dynamics_phase_weight", 0.25)),
                return_step_mask=bool(obs.get("return_step_mask", False)),
                verify_seed=obs.get("verify_seed"),
            )
            distances = verify_result.pop("distances")
            tau_timesteps = verify_result.pop("tau_timesteps")
            return {
                **verify_result,
                "tau_timesteps": tau_timesteps.numpy().tolist(),
                "distance_mean": float(distances.mean().item()),
                "distance_max": float(distances.max().item()),
                "distance_p95": float(torch.quantile(distances.flatten(), 0.95).item()),
                "distance_by_tau_mean": distances.mean(dim=(1, 2)).numpy().tolist(),
            }
        elif verify_world_latent:
            logger.info(f"################# Verify World Latent Chunk #################")
            draft_latent = obs.get("video_latent")
            if draft_latent is None:
                raise ValueError("verify_world_latent requires video_latent")
            if not isinstance(draft_latent, torch.Tensor):
                draft_latent = torch.as_tensor(draft_latent)
            if draft_latent.ndim == 4:
                draft_latent = draft_latent.unsqueeze(0)
            draft_latent = draft_latent.to(device=self.device, dtype=self.dtype)
            frame_st_id = int(obs.get("frame_st_id", self.frame_st_id))
            verify_result = self.verify_world_latent_chunk(
                draft_latent,
                frame_st_id=frame_st_id,
                tau_timesteps=obs.get("tau_timesteps", (150.0, 300.0)),
                threshold=float(obs.get("threshold", 0.35)),
            )
            distances = verify_result.pop("world_distances")
            tau_timesteps = verify_result.pop("world_tau_timesteps")
            return {
                **verify_result,
                "world_tau_timesteps": tau_timesteps.numpy().tolist(),
                "world_distance_mean": float(distances.mean().item()),
                "world_distance_max": float(distances.max().item()),
                "world_distance_p95": float(torch.quantile(distances.flatten(), 0.95).item()),
                "world_distance_by_tau_mean": distances.mean(dim=tuple(range(1, distances.ndim))).numpy().tolist(),
            }
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")
            try:
                action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            except RuntimeError as exc:
                if not _is_wanvae_temporal_cache_error(exc):
                    raise
                # ponytail: streaming WanVAE can carry a 1-frame temporal cache
                # across trial boundaries; reset once, then treat current obs as
                # a fresh prime. If this repeats, the real bug is upstream reset.
                logger.warning("WanVAE temporal cache too short; reset and retry current obs once")
                self._reset(prompt=prompt)
                action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            ret = dict(action=action)
            if obs.get("return_action_latent", False):
                ret["action_latent"] = self.last_action_latent
            if obs.get("return_video_latent", False):
                ret["video_latent"] = self.last_video_latent
            return ret
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
