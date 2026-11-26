import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import polars as pl
import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "clustering_final_optimization"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

def load_data():
    rfm = pd.read_csv(ROOT / "outputs" / "archetype_rfm.csv")
    own = pd.DataFrame(pl.read_parquet(ROOT / "outputs" / "archetype_product_ownership_features.parquet").to_dicts())
    own_cols = [c for c in own.columns if c.startswith("owns_")]
    if "n_families_owned" not in own: own["n_families_owned"] = own[own_cols].sum(axis=1)
    rfm = rfm.merge(own[["client_id", "n_families_owned"]], on="client_id", how="left")
    rfm["customer_class"] = rfm["segment"].replace({"PME":"Entreprise","TPE":"Entreprise","Grande_Entreprise":"Entreprise"})
    rfm["customer_class"] = rfm["customer_class"].where(rfm["customer_class"].eq("Entreprise"), "Retail")
    return rfm

rfm = load_data()
FEATURES = ["Monetary","Frequency","avg_balance","net_cash_flow","channel_entropy","balance_volatility","positive_cashflow_month_share","deposit_active_month_share","digital_tx_pct","atm_tx_pct","deposit_cv","low_balance_month_share","n_families_owned"]
FEATURES = [f for f in FEATURES if f in rfm]
FIN_FEATURES = [f for f in FEATURES if f != "n_families_owned"]
LOG_FEATURES = ["Monetary","avg_balance","balance_volatility","channel_entropy","digital_tx_pct","atm_tx_pct","low_balance_month_share","n_families_owned"]

def split_outliers(df, features):
    s=df[features].rank(pct=True).mean(axis=1); cut=s.quantile(.99)
    return df[s < cut].copy(), df[s >= cut].copy()

def make_matrix(df, features, pca_threshold):
    work=df.sample(min(len(df),3000),random_state=42).copy()
    for c in LOG_FEATURES:
        if c in work:
            shift=max(0,-work[c].min()+1e-6); work[c]=np.log1p(work[c]+shift)
    work=work.replace([np.inf,-np.inf],np.nan).fillna(work.median(numeric_only=True))
    used=[f for f in features if f in work and work[f].nunique()>1]
    corr=work[used].corr(method="spearman").abs(); keep=[]
    # Deterministic, archetype-independent priority: earlier features are retained.
    for f in used:
        if all(pd.isna(corr.loc[f,k]) or corr.loc[f,k] < .85 for k in keep): keep.append(f)
    X=RobustScaler().fit_transform(work[keep]).astype(float)
    n_components=np.nan
    if pca_threshold is not None:
        full=PCA(random_state=42).fit(X); n=max(2,int(np.searchsorted(np.cumsum(full.explained_variance_ratio_),pca_threshold)+1)); n=min(n,X.shape[1]); X=PCA(n_components=n,random_state=42).fit_transform(X); n_components=n
    return X,keep,n_components

def eval_labels(X,y):
    y=np.asarray(y); mask=y!=-1; Xc=X[mask]; yc=y[mask]; k=len(np.unique(yc))
    if len(Xc)<10 or k<2: return None
    counts=pd.Series(yc).value_counts()
    return dict(n_clusters=int(k),silhouette=float(silhouette_score(Xc,yc)),davies_bouldin=float(davies_bouldin_score(Xc,yc)),calinski_harabasz=float(calinski_harabasz_score(Xc,yc)),noise_ratio=float(np.mean(y==-1)),coverage=float(mask.mean()),min_cluster_size=int(counts.min()),min_cluster_fraction=float(counts.min()/len(yc)),largest_cluster=int(counts.max()),median_cluster_size=float(counts.median()))

def gmm_eval(X,k,cov,seed=42):
    model=GaussianMixture(n_components=k,covariance_type=cov,n_init=10,reg_covar=1e-3,max_iter=300,random_state=seed); y=model.fit_predict(X); z=eval_labels(X,y); z.update(bic=float(model.bic(X)),aic=float(model.aic(X))); return z

def gmm_stability(X,k,cov):
    ys=[]; sil=[]
    for seed in [1,7,42,99,123]:
        m=GaussianMixture(n_components=k,covariance_type=cov,n_init=5,reg_covar=1e-3,max_iter=300,random_state=seed); y=m.fit_predict(X); ys.append(y); sil.append(silhouette_score(X,y))
    ari=[adjusted_rand_score(ys[i],ys[j]) for i in range(len(ys)) for j in range(i+1,len(ys))]
    return float(np.mean(sil)),float(np.std(sil)),float(np.mean(ari)),float(np.std(ari))

rows=[]
STABILITY_ENABLED=False
for pop in ["Retail","Entreprise"]:
    data=rfm[rfm.customer_class.eq(pop)]; normal,outliers=split_outliers(data,FEATURES)
    variants=[("clean_with_product_count",FEATURES),("clean_without_product_count",FIN_FEATURES)]
    for variant,fs in variants:
        for pca_t in ([.95,None] if variant=="clean_with_product_count" else [.95]):
            X,used,ncomp=make_matrix(normal,fs,pca_t)
            ks=[4,5,6] if pop=="Retail" else [2,3]
            covariances=["diag","tied","full"] if (variant=="clean_with_product_count" and pca_t==.95) else ["diag"]
            for cov in covariances:
                for k in ks:
                    try:
                        STABILITY_ENABLED=(variant=="clean_with_product_count" and pca_t==.95 and cov=="diag" and ((pop=="Retail" and k==6) or (pop=="Entreprise" and k==3)))
                        z=gmm_eval(X,k,cov); sm,ss,am,aas=gmm_stability(X,k,cov) if STABILITY_ENABLED else (np.nan,np.nan,np.nan,np.nan)
                        z.update(population=pop,candidate=f"{variant}_{'no_pca' if pca_t is None else 'pca_'+str(pca_t)}",feature_set=variant,n_features=len(used),transformation="current_log_features",deduplication="unsupervised_priority_order",pca_enabled=pca_t is not None,pca_threshold=pca_t,n_components=ncomp,model="GMM",model_parameters=f"covariance_type={cov}, n_init=10, reg_covar=1e-3",k=k,bic=z["bic"],aic=z["aic"],stability_silhouette_mean=sm,stability_silhouette_std=ss,stability_ari_mean=am,stability_ari_std=aas,validity="VALID" if (pop!="Retail" or z["min_cluster_fraction"]>=.01) else "INVALID_TINY_CLUSTER",notes=f"Fixed 3,000-row diagnostic sample; outliers excluded before fit ({len(outliers)} rows)"); rows.append(z)
                    except Exception: pass
            # One local HDBSCAN set for the baseline 95% PCA representation.
            if variant=="clean_with_product_count" and pca_t==.95:
                for mcs in [100,150,200]:
                    for ms in [15]:
                        try:
                            model=hdbscan.HDBSCAN(min_cluster_size=mcs,min_samples=ms,prediction_data=True); y=model.fit_predict(X); z=eval_labels(X,y)
                            if z is not None:
                                z.update(population=pop,candidate=f"hdbscan_mcs{mcs}_ms{ms}",feature_set=variant,n_features=len(used),transformation="current_log_features",deduplication="unsupervised_priority_order",pca_enabled=True,pca_threshold=.95,n_components=ncomp,model="HDBSCAN",model_parameters=f"min_cluster_size={mcs}, min_samples={ms}, selection=eom",k=z["n_clusters"],bic=np.nan,aic=np.nan,stability_silhouette_mean=np.nan,stability_silhouette_std=np.nan,stability_ari_mean=np.nan,stability_ari_std=np.nan,validity="VALID" if pop!="Retail" or (z["min_cluster_fraction"]>=.01 and 4<=z["n_clusters"]<=6) else "INVALID_OPERATIONAL",cluster_persistence_mean=float(np.mean(model.cluster_persistence_)),notes="Density noise excluded from metrics; coverage reported"); rows.append(z)
                        except Exception: pass

res=pd.DataFrame(rows)
for c in ["population","candidate","feature_set","n_features","transformation","deduplication","pca_enabled","pca_threshold","n_components","model","model_parameters","k","n_clusters","silhouette","davies_bouldin","calinski_harabasz","bic","aic","noise_ratio","coverage","min_cluster_size","min_cluster_fraction","largest_cluster","median_cluster_size","stability_silhouette_mean","stability_silhouette_std","stability_ari_mean","stability_ari_std","cluster_persistence_mean","validity","notes"]:
    if c not in res: res[c]=np.nan
res.to_csv(OUT/"final_candidate_results.csv",index=False)

def valid_retail(): return res[(res.population=="Retail")&(res.validity=="VALID")&(res.n_clusters.between(4,6))].sort_values(["silhouette","davies_bouldin"],ascending=[False,True])
rr=valid_retail(); ee=res[(res.population=="Entreprise")&(res.model=="GMM")&(res.k.isin([2,3]))].sort_values("silhouette",ascending=False)

for pop,q in [("Retail",res[res.population=="Retail"]),("Entreprise",ee)]:
    q=q.dropna(subset=["silhouette"])
    if not q.empty:
        plt.figure(figsize=(9,5)); plt.scatter(q.k,q.silhouette,c=q.coverage,cmap="viridis"); plt.colorbar(label="Coverage"); plt.xlabel("Clusters"); plt.ylabel("Silhouette"); plt.title(f"{pop}: silhouette vs cluster count"); plt.tight_layout(); plt.savefig(FIG/f"{pop.lower()}_silhouette_vs_k.png",dpi=150); plt.close()

retail_win=rr.iloc[0] if len(rr) else res[res.population=="Retail"].sort_values("silhouette",ascending=False).iloc[0]
ent2=ee[ee.k==2].iloc[0]; ent3=ee[ee.k==3].iloc[0]
report=f'''# Final focused clustering optimization

## Scope

This experiment uses only `archetype_data`, a fixed 3,000-row diagnostic sample per population after the existing 1% pre-clustering outlier split, and an archetype-independent correlation rule. No Eta² or source/generative segment label is used for selection.

## Retail winner

{retail_win.to_dict()}

The best valid Retail candidate is selected only after rejecting tiny clusters below 1% of assigned customers. HDBSCAN candidates report assigned-point coverage and noise together with the metrics.

## Enterprise

Geometric comparison: k=2 silhouette={ent2.silhouette:.6f}, DB={ent2.davies_bouldin:.6f}; k=3 silhouette={ent3.silhouette:.6f}, DB={ent3.davies_bouldin:.6f}. Enterprise is not forced to six clusters.

## Interpretation

The Retail target of 0.45 is {'reached' if retail_win.silhouette>=.45 else 'not reached'} by the best valid candidate on this diagnostic sample; 0.50 is {'reached' if retail_win.silhouette>=.50 else 'not reached'}. HDBSCAN is not preferred solely on silhouette because its assigned-point coverage and noise ratio must remain acceptable. The experiment does not promote any model to production.

## Limitations

These are focused diagnostic results on a fixed 3,000-row sample, not a full-scale rerun of the production pipeline. HDBSCAN subsample ARI stability was not computed; its persistence is reported instead. Review finalist profiles and full-population stability before approval.
'''
(OUT/"final_candidate_comparison.md").write_text(report,encoding="utf-8")
rec=f'''# Final recommendation

## Retail

- Recommended model: {retail_win.model}
- Recommended k: {int(retail_win.k)}
- Silhouette: {retail_win.silhouette:.6f}
- DB: {retail_win.davies_bouldin:.6f}
- CH: {retail_win.calinski_harabasz:.3f}
- Coverage: {retail_win.coverage:.4%}
- Noise: {retail_win.noise_ratio:.4%}
- Smallest cluster: {int(retail_win.min_cluster_size)}
- Minimum cluster fraction: {retail_win.min_cluster_fraction:.4%}
- Stability ARI mean: {retail_win.stability_ari_mean}

Why: this is the strongest candidate remaining after the 1% tiny-cluster rejection and raw-metric/coverage checks. It is not promoted automatically if its coverage is materially lower than full-coverage GMM.

Target >=0.45 reached: {'Yes' if retail_win.silhouette>=.45 else 'No'}

Target >=0.50 reached: {'Yes' if retail_win.silhouette>=.50 else 'No'}

## Enterprise

- Best geometric solution: GMM k=2, silhouette {ent2.silhouette:.6f}, DB {ent2.davies_bouldin:.6f}
- Potential operational solution: GMM k=3, silhouette {ent3.silhouette:.6f}, DB {ent3.davies_bouldin:.6f}
- Recommendation: k=3 only if the third profile is confirmed substantial and interpretable; otherwise use k=2.

## Decision

Do not replace the current production clustering pipeline yet. This experiment remains diagnostic and requires approval after full-scale finalist validation.
'''
(OUT/"final_recommendation.md").write_text(rec,encoding="utf-8")

import nbforma
nb=nbformat.v4.new_notebook(); nb.cells=[nbformat.v4.new_markdown_cell("# Retail clustering final optimization\n\nSeparate diagnostic notebook. It uses only `archetype_data` and writes to this directory. Production notebooks and assignments are untouched."),nbformat.v4.new_code_cell(Path(__file__).read_text(encoding="utf-8"))]; nbformat.write(nb,OUT/"archetype_retail_clustering_final_optimization.ipynb")
print(res.groupby(["population","model"]).size().to_string())
print("Retail winner:", retail_win[["model","candidate","k","silhouette","davies_bouldin","coverage","noise_ratio","min_cluster_size","min_cluster_fraction","validity"]].to_dict())
print("Enterprise k2:",ent2[["silhouette","davies_bouldin","min_cluster_size","stability_ari_mean"]].to_dict())
print("Enterprise k3:",ent3[["silhouette","davies_bouldin","min_cluster_size","stability_ari_mean"]].to_dict())
