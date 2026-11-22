# Time inside real generation: compiled decode call (CPU graph/dispatch prep) and async_eval (encode).
import os, time, statistics as st, mlx.core as mx
from mlx_vlm import load, stream_generate, apply_chat_template
from exact_text_only import drop_vision
from pathlib import Path
import sys, importlib; import mlx_vlm.generate.dispatch; AR = sys.modules["mlx_vlm.generate.ar"]; print("AR", AR.__name__)
from mlx_vlm.models.inkling2 import fast_decode as FD
M=os.path.expanduser("~/.exo-inkling-bench/models/thinkingmachines--Inkling-2-Mini")
model, proc = load(M); drop_vision(model)
T={"call":[], "aeval":[], "aeval_cpu":[], "eval":[]}
oc=FD.CompiledDecoder.__call__
def c(self,*a,**k):
    t=time.perf_counter(); r=oc(self,*a,**k); T["call"].append(time.perf_counter()-t); return r
FD.CompiledDecoder.__call__=c
_ocd=FD.compiled_decode
NONE=[0]
def cd(lm,t,cache):
    r=_ocd(lm,t,cache)
    if r is None: NONE[0]+=1
    return r
FD.compiled_decode=cd
class MXW:
    def __getattr__(self, n): return getattr(mx, n)
    def async_eval(self,*a):
        t=time.perf_counter(); c0=time.thread_time(); r=mx.async_eval(*a); T["aeval"].append(time.perf_counter()-t); T.setdefault("aeval_cpu",[]).append(time.thread_time()-c0); return r
    def eval(self,*a):
        t=time.perf_counter(); r=mx.eval(*a); T["eval"].append(time.perf_counter()-t); return r
AR.mx=MXW()
tok=proc.tokenizer
CTX=int(os.environ.get("CTX","4096"))
ids=tok.encode((Path.home()/".exo-inkling-bench/bf16-longctx/corpus.txt").read_text()[:8*CTX])
doc=tok.decode(ids[:CTX-120]); cut=doc.find("\n\n",len(doc)//2)
doc=doc[:cut]+"\n\nAn archivist's note: the catalogue code of the Prague observatory archive box is VIOLET-ANCHOR-4729.\n\n"+doc[cut:]
p=apply_chat_template(proc, model.config, "Below is a long document. Read all of it, then answer the question at the end.\n\n"+doc+"\n\nQuestion: According to the archivist's note in the document above, what is the catalogue code of the Prague observatory archive box? Reply with the code only.", reasoning_effort="low")
for rep in range(2):
    for k in T: T[k].clear()
    for r in stream_generate(model, proc, p, max_tokens=256, temperature=0.0, prefill_step_size=4096): pass
    f=lambda v: "%.2f ms x%d"%(st.median(v[5:])*1e3, len(v)) if len(v)>6 else str(len(v))
    print("none",NONE[0]); NONE[0]=0
    print("rep",rep,"gen_tps %.2f"%r.generation_tps, {k:f(v) for k,v in T.items()})
