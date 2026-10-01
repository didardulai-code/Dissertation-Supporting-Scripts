import numpy as np
from scipy.stats import spearmanr
def spearman_ci(x,y,n=5000,seed=42):
    rng=np.random.default_rng(seed); x=np.asarray(x); y=np.asarray(y); k=len(x); rs=[]
    for _ in range(n):
        idx=rng.integers(0,k,k)
        if len(set(x[idx]))<3: continue
        rs.append(spearmanr(x[idx],y[idx]).correlation)
    rs=np.array(rs); r=spearmanr(x,y).correlation; p=spearmanr(x,y).pvalue
    return r,p,np.nanpercentile(rs,2.5),np.nanpercentile(rs,97.5),k
