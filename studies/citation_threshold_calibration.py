# ruff: noqa  (standalone research/repro script)
"""Citation relevance-gate calibration (litchron issue #17).

Run:  python studies/citation_threshold_calibration.py

Reference standard: BEIR/SciFact (real biomedical claim<->abstract relevance with
human qrels). We measure how well cosine similarity separates a SUPPORTING/relevant
abstract from an unrelated one, for several embedding models, and derive an
operating threshold from data instead of asserting 0.55.
"""
import os, json, math, statistics
for k in ["ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"]:
    os.environ.pop(k, None)
os.environ["HF_HOME"]="/tmp/sci-studies/hf"; os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

rng = np.random.default_rng(0)
corpus = load_dataset("BeIR/scifact","corpus",split="corpus")
queries = load_dataset("BeIR/scifact","queries",split="queries")
qrels = load_dataset("BeIR/scifact-qrels",split="test")

doc_by_id = {str(r["_id"]): (r["title"]+". "+r["text"]).strip() for r in corpus}
q_by_id   = {str(r["_id"]): r["text"].strip() for r in queries}
all_doc_ids = list(doc_by_id.keys())

# Build (query, doc, label) pairs: positives from qrels; negatives = random non-relevant docs.
rel = {}
for r in qrels:
    if int(r["score"])>=1:
        rel.setdefault(str(r["query-id"]),set()).add(str(r["corpus-id"]))
NEG_PER_POS = 5
pairs = []  # (qid, docid, label)
for qid, rels in rel.items():
    if qid not in q_by_id: continue
    for d in rels:
        if d in doc_by_id: pairs.append((qid,d,1))
    negs=set()
    while len(negs) < NEG_PER_POS*len(rels):
        cand = all_doc_ids[int(rng.integers(len(all_doc_ids)))]
        if cand not in rels: negs.add(cand)
    for d in negs: pairs.append((qid,d,0))

labels = np.array([p[2] for p in pairs])
n_pos, n_neg = int(labels.sum()), int((1-labels).sum())
print(f"pairs={len(pairs)}  pos={n_pos}  neg={n_neg}  (neg:pos = {n_neg/n_pos:.1f}:1)\n")

MODELS = {
    "all-MiniLM-L6-v2 (litchron current)": "sentence-transformers/all-MiniLM-L6-v2",
    "allenai-specter (scientific)":        "sentence-transformers/allenai-specter",
    "S-PubMedBert-MS-MARCO (biomedical)":  "pritamdeka/S-PubMedBert-MS-MARCO",
    "bge-small-en-v1.5 (strong general)":  "BAAI/bge-small-en-v1.5",
}

def boot_auc(y, s, n=1000):
    aucs=[]; idx=np.arange(len(y))
    for _ in range(n):
        b=rng.integers(0,len(y),len(y))
        if y[b].sum() in (0,len(b)): continue
        aucs.append(roc_auc_score(y[b], s[b]))
    return float(np.percentile(aucs,2.5)), float(np.percentile(aucs,97.5))

def metrics_at(y, s, t):
    pred=(s>=t).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    prec=tp/(tp+fp) if tp+fp else float('nan'); rec=tp/(tp+fn) if tp+fn else float('nan')
    return prec, rec

uniq_q = sorted(set(p[0] for p in pairs)); uniq_d = sorted(set(p[1] for p in pairs))
print(f"{'model':38s} {'AUC [95% CI]':22s} {'thr*':>5s} {'P/R@thr*':>12s} {'P/R@0.55':>12s} {'thr(P>=.95)':>11s} {'rec':>5s}")
print("-"*120)
for name, hub in MODELS.items():
    m = SentenceTransformer(hub, device="cuda")
    qemb = {qid: e for qid,e in zip(uniq_q, m.encode([q_by_id[q] for q in uniq_q], normalize_embeddings=True, batch_size=128, show_progress_bar=False))}
    demb = {did: e for did,e in zip(uniq_d, m.encode([doc_by_id[d] for d in uniq_d], normalize_embeddings=True, batch_size=128, show_progress_bar=False))}
    s = np.array([float(np.dot(qemb[q], demb[d])) for (q,d,_) in pairs])
    auc = roc_auc_score(labels, s); lo,hi = boot_auc(labels, s)
    fpr,tpr,thr = roc_curve(labels, s); j = tpr-fpr; t_star = float(thr[int(np.argmax(j))])
    p_star,r_star = metrics_at(labels,s,t_star)
    p55,r55 = metrics_at(labels,s,0.55)
    # smallest threshold achieving precision >= 0.95
    prec,rec,pthr = precision_recall_curve(labels, s)
    t95=None; r95=float('nan')
    for pp,rr,tt in zip(prec[:-1],rec[:-1],pthr):
        if pp>=0.95: t95=float(tt); r95=float(rr); break
    t95s = f"{t95:.3f}" if t95 is not None else "n/a"
    print(f"{name:38s} {auc:.3f} [{lo:.3f},{hi:.3f}]   {t_star:5.2f} {p_star:.2f}/{r_star:.2f}    {p55:.2f}/{r55:.2f}     {t95s:>11s} {r95:5.2f}")
print("\nNotes: thr* = Youden-J operating point; P/R@0.55 = litchron's current cutoff applied to each model;")
print("thr(P>=.95) = smallest cosine threshold reaching 95% precision, with the recall retained there.")
