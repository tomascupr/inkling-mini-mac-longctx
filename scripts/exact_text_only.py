"""Free the vision tower for text-only serving (no effect on text logits:
Inkling 2 only calls it when pixel_values is given)."""
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map


def drop_vision(model):
    vt = getattr(model, "vision_tower", None)
    if vt is None:
        return 0.0
    size = sum(v.nbytes for _, v in tree_flatten(vt.parameters()))
    vt.update(tree_map(lambda _: mx.zeros((0,)), vt.parameters()))
    mx.clear_cache()
    return size / 1024**3
