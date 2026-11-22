import os
os.environ["MLX_VLM_INKLING2_PREFILL_FUSED"]="1"
import mlx.core as mx
from mlx_vlm import load
from exact_text_only import drop_vision
from pathlib import Path
from mlx_vlm.models.inkling2 import prefill_fused as P
M=os.path.expanduser("~/.exo-inkling-bench/models/thinkingmachines--Inkling-2-Mini")
model, proc = load(M); drop_vision(model)
tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
lm=model.language_model
corpus=(Path.home()/".exo-inkling-bench/bf16-longctx/corpus.txt").read_text()
ids=tok.encode(corpus[:20000], add_special_tokens=False)
F=["_AR","_MG","_MOE","_SC","_L2"]
def run(n, on):
    for f in F: setattr(P,f,f in on)
    c=lm.make_cache(); o=lm(mx.array(ids[:n])[None], cache=c)
    lo=o.logits if hasattr(o,"logits") else o
    lo=lo.astype(mx.float32); mx.eval(lo); return lo
for n in [int(x) for x in os.environ.get("NS","36 100 151 645 2048").split()]:
    base=run(n,[])
    res=[]
    for f in F+["ALL"]:
        t=run(n, F if f=="ALL" else [f])
        d=mx.abs(t-base).max().item()
        res.append("%s:%s"%(f, "EQ" if d==0 else "%.3g"%d))
    print(n, " ".join(res), flush=True)
print("ISO2DONE")
