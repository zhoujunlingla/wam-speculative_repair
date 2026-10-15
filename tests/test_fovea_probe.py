import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.robotwin.fovea_probe import (  # noqa: E402
    FoveaProbeClientPolicy,
    chunk_translation_speed,
    classify_chunk,
    degrade_image,
)
from scripts.analyze_fovea_probe import mcnemar_exact_p, summarize  # noqa: E402

HEAD = "observation.images.cam_high"
LEFT = "observation.images.cam_left_wrist"
RIGHT = "observation.images.cam_right_wrist"


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8) for k in (HEAD, LEFT, RIGHT)}


def _chunk(grip_switch=False, step=0.01):
    action = np.zeros((16, 2, 16), dtype=np.float32)
    t = np.arange(32, dtype=np.float32).reshape(2, 16) * step
    action[0] = t
    action[8] = t
    if grip_switch:
        action[7, 1, 8:] = 1.0
    return action


class _Recorder:
    def __init__(self):
        self.calls = []

    def infer(self, obs):
        self.calls.append(obs)
        return {"action": np.zeros((16, 2, 16))}


def test_degrade_keeps_shape_and_removes_detail():
    img = _frame()[HEAD]
    out = degrade_image(img, 4)
    assert out.shape == img.shape and out.dtype == img.dtype
    hf = lambda x: np.abs(np.diff(x.astype(np.float32), axis=1)).mean()
    assert hf(out) < 0.5 * hf(img)
    assert degrade_image(img, 1) is img


def test_phase_rules():
    assert classify_chunk(_chunk(grip_switch=True), "gripper", 0.003)["fine"]
    assert not classify_chunk(_chunk(), "gripper", 0.003)["fine"]
    slow = _chunk(step=0.001)
    assert chunk_translation_speed(slow) < 0.003
    assert classify_chunk(slow, "speed", 0.003)["fine"]
    assert not classify_chunk(_chunk(step=0.01), "either", 0.003)["fine"]


def test_wrapper_degrades_only_selected_views_and_phase(tmp_path):
    inner = _Recorder()
    log = tmp_path / "probe.jsonl"
    probe = FoveaProbeClientPolicy(inner, view="wrist", phase="fine", factor=4, rule="gripper", log_path=str(log))
    probe.infer({"reset": True, "prompt": "p"})
    frames = [_frame(1), _frame(2)]
    originals = [{k: v.copy() for k, v in f.items()} for f in frames]

    probe.infer({"obs": frames, "compute_kv_cache": True, "state": _chunk(grip_switch=True)})
    sent = inner.calls[-1]["obs"]
    for f, o in zip(sent, originals):
        assert np.array_equal(f[HEAD], o[HEAD])
        assert not np.array_equal(f[LEFT], o[LEFT])
        assert not np.array_equal(f[RIGHT], o[RIGHT])
    for f, o in zip(frames, originals):  # client buffers untouched
        assert all(np.array_equal(f[k], o[k]) for k in o)

    probe.infer({"obs": frames, "compute_kv_cache": True, "state": _chunk(grip_switch=False)})
    assert all(np.array_equal(f[LEFT], o[LEFT]) for f, o in zip(inner.calls[-1]["obs"], originals))

    probe.end_episode(True, seed=7)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["degraded"] for r in records if r["event"] == "chunk"] == [True, False]
    episode = [r for r in records if r["event"] == "episode"][0]
    assert episode["success"] and episode["fine_chunks"] == 1 and episode["degraded_batches"] == 1


def test_phase_all_also_degrades_initial_prediction_frame():
    inner = _Recorder()
    probe = FoveaProbeClientPolicy(inner, view="head", phase="all", factor=4)
    probe.infer({"reset": True})
    frame = _frame(3)
    probe.infer({"obs": frame, "prompt": "p"})
    assert not np.array_equal(inner.calls[-1]["obs"][HEAD], frame[HEAD])
    assert np.array_equal(inner.calls[-1]["obs"][LEFT], frame[LEFT])


def test_analysis_pairs_by_seed(tmp_path):
    task_dir = tmp_path / "hanging_mug"
    task_dir.mkdir()
    def write(cond, results):
        with (task_dir / f"{cond}.jsonl").open("w") as f:
            for seed, ok in results.items():
                f.write(json.dumps({"event": "episode", "condition": cond, "episode": seed,
                                    "seed": seed, "success": ok}) + "\n")
    write("all-all-x1-either", {1: True, 2: True, 3: False, 4: True})
    write("wrist-fine-x4-either", {1: False, 2: True, 3: False, 4: False})
    rows = {r["condition"]: r for r in summarize(sorted(map(str, task_dir.glob("*.jsonl"))),
                                                 "all-all-x1-either")}
    probe = rows["wrist-fine-x4-either"]
    assert probe["paired_n"] == 4 and probe["broken"] == 2 and probe["fixed"] == 0
    assert abs(probe["delta"] + 0.5) < 1e-9
    assert mcnemar_exact_p(2, 0) == 0.5
