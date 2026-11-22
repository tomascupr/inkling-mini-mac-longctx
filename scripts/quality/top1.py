# Relaxed-gate harness: greedy answers + teacher-forced top-1 agreement on a fixed prompt set.
# MODE=ref : greedy 64-token continuations (writes top1-ref.json) + teacher-forced argmax
# MODE=test: teacher-forced argmax on prompt+ref continuation, greedy answers too (writes top1-<TAG>.json)
import os, json, time
import mlx.core as mx
from mlx_vlm import load, apply_chat_template, stream_generate
from exact_text_only import drop_vision
from pathlib import Path
M=os.path.expanduser("~/.exo-inkling-bench/models/thinkingmachines--Inkling-2-Mini")
model, proc = load(M); drop_vision(model)
tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
corpus=(Path.home()/".exo-inkling-bench/bf16-longctx/corpus.txt").read_text()
Q=["Summarise the passage above in three sentences.",
   "List every person named in the passage above.",
   "What is the main argument of the passage above? Answer briefly.",
   "Translate the first two sentences of the passage above into Czech.",
   "Write three quiz questions about the passage above, with answers.",
   "Which dates or years are mentioned in the passage above?"]
SIZES=[3000, 12000, 30000, 60000, 6000, 20000]
OFF=[0, 50000, 120000, 200000, 300000, 400000]
prompts=[]
for i,(q,n,o) in enumerate(zip(Q,SIZES,OFF)):
    prompts.append(apply_chat_template(proc, model.config, corpus[o:o+n]+"\n\n"+q, reasoning_effort="low"))
prompts.append(apply_chat_template(proc, model.config, "Write a Python function that returns the n-th Fibonacci number iteratively, then explain it in two sentences.", reasoning_effort="low"))
prompts.append(apply_chat_template(proc, model.config, "Give me a packing list for a three-day hiking trip in the Alps in October.", reasoning_effort="low"))
MODE=os.environ.get("MODE","ref"); TAG=os.environ.get("TAG","test"); N=64; STEP=int(os.environ.get("STEP","2048"))
D=Path(os.environ.get("OUTDIR","."))
ref=json.load(open(D/"top1-ref.json")) if MODE=="test" else None
lm=model.language_model
def tf_argmax(ids, start):
    """argmax of next-token logits at positions start-1 .. len-2 (predicting ids[start:])."""
    cache=lm.make_cache(); out=[]
    for s in range(0, len(ids), STEP):
        chunk=mx.array(ids[s:s+STEP])[None]
        o=lm(chunk, cache=cache)
        logits=o.logits if hasattr(o,"logits") else o
        lo=max(start-1, s); hi=min(len(ids)-1, s+chunk.shape[1])
        if hi>lo:
            out+=mx.argmax(logits[0, lo-s:hi-s], axis=-1).tolist()
        mx.eval([c.state for c in cache if hasattr(c,"state")] if False else [])
        del logits, o
    return out
res=[]
for i,p in enumerate(prompts):
    pids=tok.encode(p, add_special_tokens=False) if hasattr(tok,"encode") else None
    gen=[]
    if os.environ.get("SKIP_GREEDY")=="1":
        class _R: pass
        r=_R(); r.prompt_tokens=len(pids)
    else:
        for r in stream_generate(model, proc, p, max_tokens=N, temperature=0.0, prefill_step_size=STEP):
            gen.append(r.token)
    rec={"i":i,"prompt_tokens":r.prompt_tokens,"greedy":gen}
    cont = gen if MODE=="ref" else ref["runs"][i]["greedy"]
    if pids is not None and len(pids)==r.prompt_tokens:
        rec["tf_argmax"]=tf_argmax(pids+cont, len(pids))
        rec["tf_target"]=cont
    else:
        rec["tf_error"]="prompt token count mismatch %s vs %s"%(len(pids) if pids else None, r.prompt_tokens)
    res.append(rec); print(i, r.prompt_tokens, "greedy_eq_ref" if MODE=="test" and gen==ref["runs"][i]["greedy"] else "", flush=True)
json.dump({"mode":MODE,"tag":TAG,"env":{k:v for k,v in os.environ.items() if k.startswith(("MLX","PF_"))},"runs":res}, open(D/("top1-ref.json" if MODE=="ref" else f"top1-{TAG}.json"),"w"))
if MODE=="test":
    tot=agree=0; geq=0
    for a,b in zip(res, ref["runs"]):
        geq+= a["greedy"]==b["greedy"]
        if "tf_argmax" in a:
            t=a["tf_target"]; x=a["tf_argmax"]; tot+=len(t); agree+=sum(u==v for u,v in zip(x,t))
    print("SUMMARY greedy_identical %d/%d  teacher-forced top1 vs fork greedy %d/%d = %.2f%%"%(geq,len(res),agree,tot,100*agree/max(tot,1)))
    tot=agree=0
    for a,b in zip(res, ref["runs"]):
        if "tf_argmax" in a and "tf_argmax" in b:
            tot+=len(a["tf_argmax"]); agree+=sum(u==v for u,v in zip(a["tf_argmax"],b["tf_argmax"]))
    print("SUMMARY teacher-forced top1 vs fork teacher-forced %d/%d = %.2f%%"%(agree,tot,100*agree/max(tot,1)))
