"""Inkling 2 Mini prefix cache: follow-ups on a restored prefix, checkpoints on disk.

The model is the tiny looped one from test_inkling2_loop: 4 sliding layers with
a 4-key ring, 2 global layers, the loop core run 3 times (10 cache entries, each
a K/V cache plus its convolution state). Everything but the two Metal tests runs
on CPU in FP32.
"""

from types import SimpleNamespace as NS

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm import apc
from mlx_vlm.generate.ar import BatchGenerator, _make_cache
from mlx_vlm.models import cache as caches
from mlx_vlm.models.inkling2 import kernels, language
from mlx_vlm.tests.test_inkling2_loop import loop_case
from mlx_vlm.utils import StoppingCriteria

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
GUARD = 8  # tokens left out of a checkpoint, as APC_CHECKPOINT_GUARD_TOKENS


def document(length=300):
    mx.random.seed(0)
    return [int(t) for t in mx.random.randint(1, 59, (length,)).tolist()]


def manager(root=None):
    return apc.APCManager(
        num_blocks=16,
        block_size=16,
        disk=apc.DiskBlockStore(root, namespace="t") if root else None,
        overrides={
            "memory_max_gb": 1.0,
            "memory_reserve_gb": 0,
            "checkpoint_entries": 1,
            "checkpoint_interval_tokens": 0,
            "checkpoint_guard_tokens": GUARD,
        },
    )


def serve(lm, tokens, apc_manager, max_tokens=6):
    """One request through the server's scheduler: (tokens, logprobs, cached)."""
    runner = BatchGenerator(
        lm,
        NS(stopping_criteria=StoppingCriteria([])),
        max_tokens=max_tokens,
        prefill_batch_size=1,
        completion_batch_size=1,
        prefill_step_size=16,
        apc_manager=apc_manager,
        manage_wired_limit=False,
    )
    try:
        # As the server does: embeddings, and the request's media hash (text only).
        embeds = lm.model.embed(mx.array([tokens]))
        kwargs = {"inputs_embeds": embeds, "_apc_semantic_hash": 0}
        runner.insert([tokens], prompt_kwargs=[kwargs])
        out, logprobs, cached = [], [], 0
        while runner.has_work:
            progress, results = runner.next()
            cached = max([cached] + [p.cached_tokens for p in progress])
            for r in results:
                if r.token is not None:
                    out.append(r.token)
                    logprobs.append(r.token_logprob)
        return out, np.array(logprobs), cached
    finally:
        runner.close()


def reference_attention(q, k, v, rel, sinks, *, scale, window, key_start):
    """What kernels.relative_attention computes, in plain MLX."""
    (B, Hq, L, _), S, R = q.shape, k.shape[2], rel.shape[-1]
    k, v = (mx.repeat(a, Hq // a.shape[1], axis=1) for a in (k, v))
    keys = mx.arange(S)
    distance = (S - L + mx.arange(L))[:, None] - keys[None]
    bias = mx.take_along_axis(
        rel.transpose(0, 2, 1, 3),
        mx.broadcast_to(mx.clip(distance, 0, R - 1), (B, Hq, L, S)),
        axis=-1,
    )
    visible = (distance >= 0) & (keys >= key_start.reshape(B, 1, 1, 1))
    if window:
        visible = visible & (distance < window)
    scores = (q @ k.swapaxes(-1, -2)) * scale + mx.where(distance < R, bias, 0)
    scores = mx.where(visible, scores, -mx.inf)
    scores = mx.concatenate([scores, sinks.transpose(0, 2, 1)[..., None]], axis=-1)
    return (mx.softmax(scores, axis=-1)[..., :S] @ v).transpose(0, 2, 1, 3)


def kernel_only(monkeypatch):
    """Send every prefill chunk through the kernel's contract and record it."""
    calls = []

    def attention(q, k, v, rel, sinks, **kwargs):
        calls.append((q.shape[2], k.shape[2], kwargs["key_start"].tolist()))
        return reference_attention(q, k, v, rel, sinks, **kwargs)

    def dense_mask(*args, **kwargs):
        raise AssertionError("a prefill chunk built the heads x chunk x keys mask")

    monkeypatch.setattr(language, "_prefill_kernel", lambda q, k, v: True)
    monkeypatch.setattr(kernels, "relative_attention", attention)
    monkeypatch.setattr(language, "banded_additive_mask", dense_mask)
    return calls


def batch_cache_prefill(lm, ids, padding):
    """Prefill one left-padded row on the server's batch caches, in three chunks."""
    cache = _make_cache(lm, [padding])
    prompt = mx.concatenate([mx.zeros((1, padding), mx.int32), ids[:, :9]], axis=1)
    parts = [lm(prompt, cache=cache).logits[:, padding:]]
    parts += [lm(ids[:, a:b], cache=cache).logits for a, b in ((9, 15), (15, 21))]
    return mx.concatenate(parts, axis=1).astype(mx.float32)


def test_follow_ups_restore_the_conversation_and_match_cold():
    lm = loop_case()[0].language_model
    prompt, cache, checkpoint = document(), manager(), 0
    for _ in range(3):
        expected, expected_logprobs, _ = serve(lm, prompt, None)
        answer, logprobs, cached = serve(lm, prompt, cache)
        assert cached == checkpoint
        assert answer == expected
        np.testing.assert_allclose(logprobs, expected_logprobs, atol=1e-5)
        checkpoint = len(prompt) - GUARD
        prompt = prompt + answer + [7, 8, 9, 10, 11]
    assert cache.stats.exact_hits == 2 and cache.stats.memory_skips == 0


def test_follow_up_prefill_stays_in_the_kernel(monkeypatch):
    # A follow-up prefills its new tokens on one-row batch caches that hold the
    # restored prefix. It must not fall back to the dense mask, which at 131k
    # keys is 24 heads x chunk x 131k per global layer pass.
    lm = loop_case()[0].language_model
    prompt, cache = document(), manager()
    answer, _, _ = serve(lm, prompt, cache)
    follow = prompt + answer + [7 + i % 50 for i in range(40)]
    expected, expected_logprobs, _ = serve(lm, follow, None)

    calls = kernel_only(monkeypatch)
    got, logprobs, cached = serve(lm, follow, cache)
    assert cached == len(prompt) - GUARD
    # Global layers attend to the whole restored prefix, from its first key.
    # (The last prompt token is the first decode step.)
    assert max(keys for _, keys, _ in calls) == len(follow) - 1
    assert {tuple(start) for _, _, start in calls} == {(0,)}
    assert got == expected
    np.testing.assert_allclose(logprobs, expected_logprobs, atol=1e-4)


@pytest.mark.parametrize("padding", [0, 3])
def test_batch_cache_prefill_through_the_kernel_contract(padding, monkeypatch):
    # Left padding is each row's first key; the sliding ring trims it away.
    lm = loop_case()[0].language_model
    ids = mx.array([list(range(1, 22))])
    expected = np.array(lm(ids).logits)
    calls = kernel_only(monkeypatch)
    actual = batch_cache_prefill(lm, ids, padding)
    assert {start[0] for _, _, start in calls} == {0, padding}
    np.testing.assert_allclose(np.array(actual), expected, atol=3e-5)


@metal
@pytest.mark.parametrize("padding", [0, 3])
def test_bf16_batch_cache_prefill_kernel_matches_dense_mask(padding, monkeypatch):
    # The same on Metal in BF16: the kernel against the dense mask it replaces.
    lm = loop_case(mx.bfloat16)[0].language_model
    ids = mx.array([list(range(1, 22))])
    with monkeypatch.context() as dense:
        dense.setattr(language, "_prefill_kernel", lambda q, k, v: False)
        expected = batch_cache_prefill(lm, ids, padding)

    def dense_mask(*args, **kwargs):
        raise AssertionError("a prefill chunk built the heads x chunk x keys mask")

    monkeypatch.setattr(language, "banded_additive_mask", dense_mask)
    actual = batch_cache_prefill(lm, ids, padding)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=5e-2, rtol=5e-2
    )


@metal
@pytest.mark.parametrize(
    "keys,window,start",
    [(5000, 0, 0), (5000, 0, 700), (5000, 0, 4990), (1063, 1024, 500)],
)
def test_relative_attention_kernel_first_key(keys, window, start):
    # The model's head shape: a 40-token chunk whose first `start` keys are left
    # padding (whole key blocks, part of one, and past the first queries), on a
    # global layer and on a sliding layer's ring.
    mx.random.seed(11)
    hq, hkv, d, r, length = 24, 8, 128, 1024, 40
    q = mx.random.normal((1, hq, length, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, hkv, keys, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, keys, d)).astype(mx.bfloat16)
    rel = mx.random.normal((1, length, hq, r)).astype(mx.bfloat16)
    sinks = mx.random.normal((1, length, hq)).astype(mx.bfloat16)
    args = dict(scale=d**-0.5, window=window, key_start=mx.array([start]))
    actual = kernels.relative_attention(q, k, v, rel, sinks, **args)
    expected = reference_attention(
        *(a.astype(mx.float32) for a in (q, k, v, rel, sinks)), **args
    )
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), np.array(expected), atol=1e-2, rtol=1e-2
    )


def leaves(cache):
    """Every K/V cache and convolution state of a prompt cache, in order."""
    for entry in cache:
        if isinstance(entry, caches.CacheList):
            yield from leaves(entry.caches)
        else:
            yield entry


def assert_same_state(actual, expected):
    actual, expected = list(leaves(actual)), list(leaves(expected))
    assert [type(c) for c in actual] == [type(c) for c in expected]
    for a, e in zip(actual, expected):
        if isinstance(e, caches.ArraysCache):
            pairs = zip(a.cache, e.cache, strict=True)
        else:
            assert (
                a.offset,
                getattr(a, "_idx", None),
                getattr(a, "max_size", None),
            ) == (
                e.offset,
                getattr(e, "_idx", None),
                getattr(e, "max_size", None),
            )
            pairs = zip(a.state, e.state, strict=True)
        for x, y in pairs:
            assert (x is None) == (y is None)
            if y is not None:
                assert x.dtype == y.dtype and x.shape == y.shape
                assert mx.array_equal(x.astype(mx.float32), y.astype(mx.float32))


def reopen(root, key):
    """A restart: a new store finds the checkpoint from the files alone."""
    store = apc.DiskBlockStore(root, namespace="t")
    try:
        return store.load_exact_cache(key)
    finally:
        store.close()


def test_checkpoint_on_disk_restores_exact_logits(tmp_path):
    lm = loop_case()[0].language_model
    ids = mx.array([document(48)])

    def prefill(cache):
        # Chunks trim the 4-key ring; single tokens then rotate it in place.
        for start in range(0, 32, 8):
            mx.eval(lm(ids[:, start : start + 8], cache=cache).logits)
        for i in range(32, 35):
            mx.eval(lm(ids[:, i : i + 1], cache=cache).logits)
        return cache

    def continuation(cache):
        parts = [lm(ids[:, 35:41], cache=cache).logits]
        parts += [lm(ids[:, i : i + 1], cache=cache).logits for i in range(41, 48)]
        return np.array(mx.concatenate(parts, axis=1))

    live = prefill(lm.make_cache())
    assert len(live) == 10 and live[0][0]._idx not in (0, live[0][0].max_size)
    store = apc.DiskBlockStore(tmp_path, namespace="t")
    assert store.save_exact_cache(1, ids[0, :35].tolist(), 0, live, synchronous=True)
    store.close()
    assert [p.suffix for p in tmp_path.rglob("exact_*")] == [".safetensors"]

    tokens, _, restored = reopen(tmp_path, 1)
    assert list(tokens) == ids[0, :35].tolist()
    assert_same_state(restored, live)
    logits = continuation(restored)
    assert np.array_equal(logits, continuation(prefill(lm.make_cache())))
    np.testing.assert_allclose(logits, np.array(lm(ids).logits[:, 35:]), atol=3e-5)


def test_bf16_checkpoint_round_trips_bit_for_bit(tmp_path):
    # The model's K/V is BF16 and its convolution state FP32; a ring that has
    # wrapped keeps its raw slot order and write index.
    mx.random.seed(1)
    kv = caches.KVCache()
    kv.keys, kv.values = (
        mx.random.normal((1, 2, 512, 8)).astype(mx.bfloat16) for _ in range(2)
    )
    kv.offset = 300
    ring = caches.RotatingKVCache(max_size=16)
    ring.keys, ring.values = (
        mx.random.normal((1, 2, 16, 8)).astype(mx.bfloat16) for _ in range(2)
    )
    ring.offset, ring._idx = 1000, 5
    conv = caches.ArraysCache(4)
    conv[0], conv[2] = mx.random.normal((1, 3, 8)), mx.random.normal((1, 3, 16))
    saved = [caches.CacheList(kv, conv), caches.CacheList(ring, caches.ArraysCache(4))]
    store = apc.DiskBlockStore(tmp_path, namespace="t")
    assert store.save_exact_cache(2, list(range(300)), 0, saved, synchronous=True)
    store.close()

    _, _, restored = reopen(tmp_path, 2)
    assert_same_state(restored, saved)
    assert restored[0][0].keys.dtype == restored[1][0].values.dtype == mx.bfloat16
    assert restored[0][1][0].dtype == mx.float32 and restored[0][1][1] is None


def test_follow_up_after_a_restart_restores_from_disk(tmp_path):
    lm = loop_case()[0].language_model
    prompt = document()
    first = manager(tmp_path)
    answer, _, _ = serve(lm, prompt, first)
    first.disk.close()

    follow = prompt + answer + [7, 8, 9, 10, 11]
    expected, expected_logprobs, _ = serve(lm, follow, None)
    restarted = manager(tmp_path)
    try:
        got, logprobs, cached = serve(lm, follow, restarted)
    finally:
        restarted.disk.close()
    assert cached == len(prompt) - GUARD and restarted.stats.disk_hits == 1
    assert got == expected
    np.testing.assert_allclose(logprobs, expected_logprobs, atol=1e-5)
