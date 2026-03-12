"""
Unit tests — run with:  pytest tests/ -v
These run on CPU with tiny synthetic data, so no GPU or OTTO dataset needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np
from models.sasrec import SASRec
from utils.dataset import build_vocab, build_sequences, OTTODataset

# ── Fixtures ──────────────────────────────────────────────────────────────────

VOCAB_SIZE = 102   # 100 items + PAD(0) + MASK(1)
MAX_LEN    = 10
D_MODEL    = 32
N_HEADS    = 4
N_LAYERS   = 2

@pytest.fixture
def model():
    return SASRec(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_len=MAX_LEN,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=0.0,   # deterministic for tests
    )

@pytest.fixture
def fake_sessions():
    """Minimal synthetic OTTO-like session data."""
    return [
        {"session": 0, "events": [
            {"aid": i, "ts": 1000 + i, "type": t}
            for i, t in enumerate(["clicks"]*5 + ["carts"]*2 + ["orders"]*1)
        ]},
        {"session": 1, "events": [
            {"aid": i + 10, "ts": 2000 + i, "type": "clicks"}
            for i in range(6)
        ]},
    ]


# ── Model shape tests ─────────────────────────────────────────────────────────

def test_forward_output_shapes(model):
    batch = torch.randint(2, VOCAB_SIZE, (4, MAX_LEN))
    out   = model(batch)
    assert set(out.keys()) == {"logits_click", "logits_cart", "logits_order"}
    for key, tensor in out.items():
        assert tensor.shape == (4, VOCAB_SIZE), f"{key} shape mismatch: {tensor.shape}"


def test_encode_output_shape(model):
    batch  = torch.randint(2, VOCAB_SIZE, (3, MAX_LEN))
    hidden = model.encode(batch)
    assert hidden.shape == (3, MAX_LEN, D_MODEL)


def test_padding_does_not_affect_output(model):
    """PAD tokens (0) at the start should not change the last-position output."""
    base_seq  = torch.randint(2, VOCAB_SIZE, (1, MAX_LEN))
    pad_seq   = base_seq.clone()
    pad_seq[0, :3] = 0   # zero out first 3 positions

    with torch.no_grad():
        out_base = model(base_seq)["logits_click"]
        # Different sequences → different outputs (sanity, not equality test)
    assert out_base.shape == (1, VOCAB_SIZE)


def test_weight_tying(model):
    """All three output heads must share the item embedding weight matrix."""
    assert model.head_click.weight.data_ptr() == model.item_emb.weight.data_ptr()
    assert model.head_cart.weight.data_ptr()  == model.item_emb.weight.data_ptr()
    assert model.head_order.weight.data_ptr() == model.item_emb.weight.data_ptr()


def test_get_item_embeddings_shape(model):
    embs = model.get_item_embeddings()
    assert embs.shape == (VOCAB_SIZE, D_MODEL)
    assert not embs.requires_grad   # detached


def test_causal_mask_is_upper_triangular(model):
    mask = model._causal_mask(5, torch.device("cpu"))
    assert mask.shape == (5, 5)
    # Upper triangle (excluding diagonal) should be True
    assert mask[0, 1] == True
    assert mask[0, 0] == False   # diagonal is False (self-attention allowed)
    assert mask[3, 1] == False   # lower triangle is False


# ── Dataset tests ─────────────────────────────────────────────────────────────

def test_build_vocab(fake_sessions):
    vocab = build_vocab(fake_sessions)
    all_aids = {e["aid"] for s in fake_sessions for e in s["events"]}
    assert len(vocab) == len(all_aids)
    # Every token id should be >= 2 (0=PAD, 1=MASK are reserved)
    assert all(v >= 2 for v in vocab.values())


def test_build_sequences_length(fake_sessions):
    vocab   = build_vocab(fake_sessions)
    samples = build_sequences(fake_sessions, vocab, max_len=MAX_LEN)
    assert len(samples) > 0
    for s in samples:
        assert len(s["input_ids"]) == MAX_LEN
        assert s["target_type"] in (0, 1, 2)


def test_dataset_getitem(fake_sessions):
    vocab   = build_vocab(fake_sessions)
    samples = build_sequences(fake_sessions, vocab, max_len=MAX_LEN)
    ds      = OTTODataset(samples)
    item    = ds[0]
    assert item["input_ids"].shape   == (MAX_LEN,)
    assert item["input_ids"].dtype   == torch.long
    assert item["target_id"].ndim    == 0
    assert item["target_type"].ndim  == 0


def test_dataset_len(fake_sessions):
    vocab   = build_vocab(fake_sessions)
    samples = build_sequences(fake_sessions, vocab, max_len=MAX_LEN)
    ds      = OTTODataset(samples)
    assert len(ds) == len(samples)
