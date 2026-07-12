from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wan_va"))

from modules.model import WanAttention  # noqa: E402


def test_readonly_attention_preserves_every_cache_tensor():
    attention = WanAttention(dim=4, heads=1, dim_head=4, attn_mode="torch")
    attention.init_kv_cache(
        "pos",
        total_tolen=4,
        num_head=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=2,
    )
    cache = attention.attn_caches["pos"]
    cache["k"].zero_()
    cache["v"].zero_()
    cache["k"][:, :2] = torch.randn_like(cache["k"][:, :2])
    cache["v"][:, :2] = torch.randn_like(cache["v"][:, :2])
    cache["id"][:2] = torch.tensor([0, 1])
    cache["mask"][:2] = True
    before = {name: tensor.clone() for name, tensor in cache.items()}

    query = torch.randn(1, 1, 4)
    output = attention(
        query,
        query,
        query,
        rotary_emb=None,
        update_cache=0,
        cache_name="pos",
        readonly_cache_indices=(0, ),
    )

    assert output.shape == query.shape
    for name, expected in before.items():
        assert torch.equal(cache[name], expected), name
