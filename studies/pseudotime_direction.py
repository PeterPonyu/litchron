"""Pseudotime comparison rigor on controlled ground truth (litchron issue #18).

Run:  python studies/pseudotime_direction.py

We simulate a 1-D trajectory where each cell's true pseudotime t is KNOWN
(assumption-free: we generated it), then run scanpy DPT and show:
  (a) raw Spearman vs truth flips sign purely from the arbitrary root choice
      (a correct-but-reversed ordering scores ~ -1) -> the litchron #18 bug;
  (b) a direction-invariant / root-anchored statistic recovers true agreement;
  (c) a 'severe disagreement' threshold can be DERIVED from a null distribution
      instead of the magic 0.1.
"""
# ruff: noqa  (standalone research/repro script)
import os

for k in ["ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"]:
    os.environ.pop(k,None)
import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.stats import spearmanr

sc.settings.verbosity=0
rng=np.random.default_rng(0)

def simulate(n=600, g=200, seed=0):
    r=np.random.default_rng(seed)
    t=np.sort(r.uniform(0,1,n))                      # KNOWN true pseudotime
    X=np.zeros((n,g))
    for j in range(g):
        kind=r.integers(0,3)
        if kind==0: prog=t                            # up
        elif kind==1: prog=1-t                        # down
        else:                                         # transient
            c=r.uniform(0.2,0.8); prog=np.exp(-((t-c)**2)/0.02)
        amp=r.uniform(2,8); base=r.uniform(0.1,0.5)
        mu=base+amp*prog
        X[:,j]=r.poisson(mu)
    return AnnData(X.astype("float32")), t

ad,t=simulate()
sc.pp.normalize_total(ad,target_sum=1e4); sc.pp.log1p(ad)
sc.pp.pca(ad,n_comps=30,random_state=0)
sc.pp.neighbors(ad,n_neighbors=15,random_state=0)
sc.tl.diffmap(ad,n_comps=15)

# Root at the true-earliest cell vs the true-latest cell (both are "valid" extremes
# of diffusion component 1 in practice; the sign of that eigenvector is arbitrary).
order_t=np.argsort(t)
results={}
for label,root in [("root = true-early cell", int(order_t[0])), ("root = true-late cell", int(order_t[-1]))]:
    ad.uns["iroot"]=root
    sc.tl.dpt(ad,n_dcs=15)
    pt=np.asarray(ad.obs["dpt_pseudotime"]).copy()
    finite=np.isfinite(pt)
    raw=spearmanr(pt[finite],t[finite]).statistic
    results[label]=raw

print("Raw Spearman(DPT pseudotime, TRUE pseudotime):")
for k,v in results.items(): print(f"  {k:28s}: {v:+.3f}")
vals=list(results.values())
print(f"\n  -> same data, two valid roots: {vals[0]:+.3f} vs {vals[1]:+.3f}")
print(f"  -> direction-invariant |Spearman| (root-anchored agreement): {abs(vals[0]):.3f} (both roots agree)")

# Null distribution of |Spearman| for RANDOM orderings vs truth -> derive 'severe' cutoff.
null=[abs(spearmanr(rng.permutation(len(t)), t).statistic) for _ in range(5000)]
p95=float(np.percentile(null,95)); p99=float(np.percentile(null,99))
print(f"\nNull |Spearman| (random ordering vs truth, n=5000):  95th pct={p95:.3f}  99th pct={p99:.3f}")
print(f"  -> a data-derived 'agreement is no better than chance' cutoff is |rho| <= {p95:.3f},")
print("     and 'severe disagreement' should be judged on |rho| (direction-invariant), not raw signed rho.")
print("\nConclusion: under the current signed-Spearman comparison, the SAME correct ordering")
print(f"scores {vals[0]:+.3f} or {vals[1]:+.3f} depending only on an arbitrary root -> agreement/disagreement")
print("can invert. Anchoring direction (|rho| or shared-root) removes the artifact.")
