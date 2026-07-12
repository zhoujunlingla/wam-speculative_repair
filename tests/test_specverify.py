from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wan_va"))

from specverify import (  # noqa: E402
    gripper_consensus_prefix,
    gripper_switch_info,
    latent_frame_motion_stats,
    longest_prefix_min_over_k,
    normalized_l2_distances,
    quantize_prefix_to_frame_boundary,
)


def test_gripper_consensus_accepts_shared_transition_and_bounds_disagreement():
    draft = torch.full((1, 30, 2, 16, 1), -1.0)
    draft[:, 28, 1, 4:] = 1.0
    reconstructed = draft.repeat(2, 1, 1, 1, 1)

    prefix, failure = gripper_consensus_prefix(
        reconstructed, draft, max_prefix=32
    )
    assert (prefix, failure) == (32, None)

    reconstructed[1, 28, 1, 7, 0] = -1.0
    prefix, failure = gripper_consensus_prefix(
        reconstructed, draft, max_prefix=32
    )
    assert (prefix, failure) == (23, 23)


def test_gripper_consensus_requires_cross_tau_probes():
    draft = torch.zeros(1, 30, 2, 16, 1)
    try:
        gripper_consensus_prefix(draft, draft, max_prefix=32)
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("K=1 must not be treated as cross-tau consensus")


def test_latent_frame_motion_stats_detects_concentrated_change():
    latents = torch.zeros(1, 4, 2, 2, 2)
    latents[:, :, 1, 0, 0] = 2

    stats = latent_frame_motion_stats(latents, top_fraction=0.25)

    assert stats["global_mean"] == 0.5
    assert stats["top_mean"] == 2.0
    assert stats["top_concentration"] == 1.0
    assert stats["top_relative"] > 1.0


def test_normalized_l2_uses_only_continuous_channels():
    draft = torch.zeros(1, 30, 2, 16, 1)
    recon = torch.zeros(2, 30, 2, 16, 1)
    recon[:, 28:30] = 100
    recon[1, 0, 1, 4, 0] = 14**0.5

    distances = normalized_l2_distances(recon, draft)

    assert distances.shape == (2, 2, 16)
    assert distances[0].max().item() == 0
    assert torch.isclose(distances[1, 1, 4], torch.tensor(1.0))


def test_prefix_is_minimum_over_all_tau_rows():
    distances = torch.zeros(2, 32)
    distances[0, 20] = 1
    distances[1, 17] = 1

    raw, by_tau, pass_by_step = longest_prefix_min_over_k(distances, 0.15)

    assert raw == 17
    assert by_tau.tolist() == [20, 17]
    assert pass_by_step[:17].all()
    assert not pass_by_step[17]
    assert quantize_prefix_to_frame_boundary(
        raw, action_per_frame=16, frame_chunk_size=2
    ) == 16


def test_gripper_switch_is_independent_of_pose_distance():
    action = torch.zeros(1, 30, 2, 16, 1)
    action[:, 28:30] = -1
    action[:, 28, 1, 4:] = 1

    switch = gripper_switch_info(action, threshold=0)

    assert switch["has_switch"] is True
    assert switch["first_index"] == 20
    assert switch["channels"] == [28]
