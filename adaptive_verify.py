"""Dependency-free predicates shared by verifier shadow and live paths."""

import math


def k1_full_accept_certificate(
    *,
    distance_max: float,
    continuous_prefix: int,
    horizon: int,
    phase_agreement: float,
    reconstructed_switch: bool,
    draft_switch: bool,
    decoded_draft_switch: bool,
    verify_threshold: float,
    certificate_threshold: float,
) -> bool:
    """Return whether one probe safely decides a full-chunk K=2 pass."""

    scalars = (
        distance_max,
        phase_agreement,
        verify_threshold,
        certificate_threshold,
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        return False
    if verify_threshold < 0 or certificate_threshold < 0 or horizon <= 0:
        return False
    return bool(
        int(continuous_prefix) == int(horizon)
        and float(distance_max) <= min(
            float(verify_threshold), float(certificate_threshold)
        )
        and float(phase_agreement) == 1.0
        and not reconstructed_switch
        and not draft_switch
        and not decoded_draft_switch
    )
