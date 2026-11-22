import re,sys
p=sys.argv[1]; s=open(p).read()
s=s.replace('_PREFILL_EVAL_EVERY = int(_os.environ.get("MLX_VLM_EXACT_PREFILL_EVAL_EVERY", "0"))',
 '_PREFILL_EVAL_EVERY = int(_os.environ.get("MLX_VLM_EXACT_PREFILL_EVAL_EVERY", "0"))\n_VISIT_TIMES = _os.environ.get("VISIT_TIMES")  # profiling only: path to append per-visit wall times\nimport time as _time',1)
old='''        for step, (i, cache_idx, p) in enumerate(self.schedule()):
            layer = self.layers[i]'''
new='''        _vt = _VISIT_TIMES and x.shape[1] >= _PREFILL_EVAL_MIN_TOKENS
        if _vt:
            mx.eval(x)
            _rows = []
            _t0 = _time.perf_counter()
        for step, (i, cache_idx, p) in enumerate(self.schedule()):
            layer = self.layers[i]'''
assert old in s; s=s.replace(old,new,1)
old2='''                (mx.async_eval if _PREFILL_EVAL_ASYNC else mx.eval)(
                    x, c.state if c is not None else []
                )
'''
new2=old2+'''                if _vt:
                    _t1 = _time.perf_counter()
                    _off = getattr(c, "offset", None) if c is not None and not isinstance(c, (list, tuple)) else None
                    _rows.append((step, i, cache_idx, p, type(c).__name__, _off, _t1 - _t0))
                    _t0 = _t1
'''
assert old2 in s; s=s.replace(old2,new2,1)
old3='''        h = self.final_input(blocks, x)[..., 0, :]'''
new3='''        if _vt:
            with open(_VISIT_TIMES, "a") as _f:
                for _r in _rows:
                    _f.write("%d\\t%d\\t%d\\t%d\\t%s\\t%s\\t%d\\t%.6f\\n" % (_r[0], _r[1], _r[2], _r[3], _r[4], _r[5], x.shape[1], _r[6]))
        h = self.final_input(blocks, x)[..., 0, :]'''
assert old3 in s; s=s.replace(old3,new3,1)
open(p,'w').write(s); print("patched")
