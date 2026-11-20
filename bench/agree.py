import json, sys, numpy as np
spearmanr=lambda x,y:(np.corrcoef(np.argsort(np.argsort(x)),np.argsort(np.argsort(y)))[0,1],)
a={json.loads(l)["row"]:json.loads(l)["p"] for l in open(sys.argv[1])}
b={json.loads(l)["row"]:json.loads(l)["p"] for l in open(sys.argv[2])}
k=sorted(set(a)&set(b)); x=np.array([a[i] for i in k]); y=np.array([b[i] for i in k])
print(f"{sys.argv[1]} vs {sys.argv[2]}: n={len(k)} max|dp|={np.abs(x-y).max():.4f} mean|dp|={np.abs(x-y).mean():.5f} spearman={spearmanr(x,y)[0]:.5f}")
