import sys
p=sys.argv[1]; s=open(p).read()
helper='''

def _exact_host_vocab(model, chunk_embed):
    """MLX_VLM_EXACT_HOST_VOCAB=1: during a chunked text prefill keep the
    vocabulary tables (embed_tokens, lm_head; 2 x 0.86 GiB bf16) in host
    memory instead of MLX buffers. MLX's Metal allocator drops its whole
    buffer cache on every cache miss once active memory nears 0.95 x the
    recommended working set (49.25 GiB here); at 1M tokens (17.6 GiB weights
    + 32 GiB KV) that made every prefill op allocate fresh buffers. The chunk
    loop never evaluates logits, and an embedding row gather is a plain copy,
    so every value is unchanged. Returns (chunk_embed, restore) or None."""
    if os.environ.get("MLX_VLM_EXACT_HOST_VOCAB", "0") != "1" or chunk_embed is None:
        return None
    import numpy as np

    lm = model.language_model
    tm = lm.model
    head = getattr(lm, "lm_head", None)
    emb = getattr(tm, "embed_tokens", None)
    if (
        type(emb) is not nn.Embedding
        or emb.weight.dtype != mx.bfloat16
        or type(head) is not nn.Linear
        or "bias" in head
        or head.weight.dtype != mx.bfloat16
    ):
        return None
    host_emb = np.array(emb.weight.view(mx.uint16))
    host_head = np.array(head.weight.view(mx.uint16))
    e_shape, h_shape = emb.weight.shape, head.weight.shape
    # Shape-only stand-ins (no storage); the chunk graphs that mention lm_head
    # are never evaluated.
    emb.weight = mx.broadcast_to(mx.zeros((1, 1), mx.bfloat16), e_shape)
    head.weight = mx.broadcast_to(mx.zeros((1, 1), mx.bfloat16), h_shape)
    mx.clear_cache()

    def host_embed(ids):
        rows = mx.array(host_emb[np.array(ids)]).view(mx.bfloat16)
        return tm.embed_norm(rows)

    def restore():
        emb.weight = mx.array(host_emb).view(mx.bfloat16)
        head.weight = mx.array(host_head).view(mx.bfloat16)
        mx.eval(emb.weight, head.weight)

    return host_embed, restore
'''
anchor="def _exact_reserve_kv(prompt_cache, new_tokens):"
assert anchor in s; s=s.replace(anchor, helper.lstrip("\n")+"\n\n"+anchor,1)
old='''        _exact_reserve_kv(prompt_cache, inputs_embeds.shape[1] + max(int(max_tokens or 0), 0))
        if prefill_step_size is not None and should_chunk:'''
new='''        _exact_reserve_kv(prompt_cache, inputs_embeds.shape[1] + max(int(max_tokens or 0), 0))
        host_vocab = _exact_host_vocab(model, chunk_embed)
        if host_vocab is not None:
            chunk_embed, restore_vocab = host_vocab
            # Drop the lazy whole-prompt embedding (it references the table);
            # only its shape is used below.
            embedding_output = None
            inputs_embeds = mx.broadcast_to(
                mx.zeros((1, 1, 1), mx.bfloat16), inputs_embeds.shape
            )
        if prefill_step_size is not None and should_chunk:'''
assert old in s; s=s.replace(old,new,1)
old2='''            input_ids = input_ids[:, -1:]
            if chunk_embed is not None:
                inputs_embeds = chunk_embed(input_ids)
'''
new2='''            input_ids = input_ids[:, -1:]
            if chunk_embed is not None:
                inputs_embeds = chunk_embed(input_ids)
        if host_vocab is not None:
            mx.eval(inputs_embeds)
            restore_vocab()
'''
assert old2 in s; s=s.replace(old2,new2,1)
if "import mlx.nn as nn" not in s and "\nimport mlx.nn as nn" not in s:
    s=s.replace("import mlx.core as mx","import mlx.core as mx\nimport mlx.nn as nn",1)
open(p,'w').write(s); print("ok")
