import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
import hdbscan

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "static_clustering" / "archetype_enriched_rfm_clustering.ipynb"
OLD = ROOT.parent / "initial_data" / "static clustering" / "notebooks" / "rfm_clustering_enriched_static.ipynb"
OUT = ROOT / "artifacts" / "clustering_ablation"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def src(nb, cell):
    return "".join(nb["cells"][cell].get("source", []))

current_nb = json.loads(CURRENT.read_text(encoding="utf-8"))
old_nb = json.loads(OLD.read_text(encoding="utf-8"))

def load_data():
    rfm = pd.read_csv(ROOT / "outputs" / "archetype_rfm.csv")
    own = pd.DataFrame(pl.read_parquet(ROOT / "outputs" / "archetype_product_ownership_features.parquet").to_dicts())
    panel = pl.read_parquet(ROOT / "outputs" / "archetype_weekly_panel.parquet", columns=["client_id", "monthly_net_cash_flow", "monthly_deposit", "ending_balance"])
    own_cols = [c for c in own.columns if c.startswith("owns_")]
    if "n_families_owned" not in own.columns:
        own["n_families_owned"] = own[own_cols].sum(axis=1)
    own_keep = ["client_id", "n_families_owned"] + [c for c in own_cols if c not in rfm.columns]
    rfm = rfm.merge(own[own_keep], on="client_id", how="left")
    enr = pd.DataFrame(panel.group_by("client_id").agg([
        (pl.col("monthly_net_cash_flow") > 0).mean().alias("positive_cashflow_week_share"),
        (pl.col("monthly_deposit") > 0).mean().alias("deposit_active_week_share"),
        (pl.col("monthly_deposit").std() / (pl.col("monthly_deposit").mean() + 1e-9)).alias("deposit_cv"),
        (pl.col("ending_balance") <= 0).mean().alias("low_balance_week_share"),
    ]).to_dicts())
    rfm = rfm.drop(columns=[c for c in enr.columns if c != "client_id" and c in rfm.columns], errors="ignore")
    rfm = rfm.merge(enr, on="client_id", how="left")
    rfm["customer_class"] = rfm["segment"].replace({"PME":"Entreprise", "TPE":"Entreprise", "Grande_Entreprise":"Entreprise"})
    rfm["customer_class"] = rfm["customer_class"].where(rfm["customer_class"].eq("Entreprise"), "Retail")
    aliases = {"positive_cashflow_week_share":"positive_cashflow_month_share", "deposit_active_week_share":"deposit_active_month_share", "low_balance_week_share":"low_balance_month_share"}
    for a, b in aliases.items(): rfm[b] = rfm[a]
    return rfm

rfm = load_data()

current_features = ["Monetary", "Frequency", "avg_balance", "net_cash_flow", "channel_entropy", "balance_volatility", "positive_cashflow_month_share", "deposit_active_month_share", "digital_tx_pct", "atm_tx_pct", "deposit_cv", "low_balance_month_share", "n_families_owned"]
old_features = ["Monetary", "Frequency", "avg_balance", "total_deposit", "total_withdrawal", "net_cash_flow", "channel_entropy", "balance_volatility", "positive_cashflow_month_share", "deposit_active_month_share", "digital_tx_pct", "longest_negative_cashflow_run", "atm_tx_pct", "deposit_cv", "low_balance_month_share", "balance_stability_ratio", "n_families_owned"]
old_features = [f for f in old_features if f in rfm.columns]
current_features = [f for f in current_features if f in rfm.columns]

def split(df, features):
    score = df[features].rank(pct=True).mean(axis=1)
    cut = score.quantile(.99)
    return df[score < cut].copy(), df[score >= cut].copy()

def transform(df, features, scheme="current"):
    out = df.copy()
    if scheme == "current":
        cols = ["Monetary", "avg_balance", "balance_volatility", "channel_entropy", "digital_tx_pct", "atm_tx_pct", "low_balance_month_share", "n_families_owned"]
    elif scheme == "old":
        cols = ["Monetary", "avg_balance", "total_deposit", "total_withdrawal", "balance_volatility", "channel_entropy", "digital_tx_pct", "longest_negative_cashflow_run", "atm_tx_pct", "low_balance_month_share", "n_families_owned"]
    else:
        cols = ["Monetary", "avg_balance", "total_deposit", "total_withdrawal", "balance_volatility"]
    for c in cols:
        if c in out:
            shift = max(0, -out[c].min() + 1e-6)
            out[c] = np.log1p(out[c] + shift)
    return ou

def matrix(df, features, scheme="current", dedup=True, pca_threshold=.95):
    if len(df) > 3000:
        df = df.sample(3000, random_state=42)
    work = transform(df, features, scheme)
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.fillna(work.median(numeric_only=True))
    used = [f for f in features if f in work and work[f].nunique() > 1]
    if not used:
        return np.empty((0, 0)), [], np.nan
    if dedup:
        keep=[]
        corr=work[used].corr(method="spearman").abs()
        for f in used:
            if all(corr.loc[f, k] < .85 for k in keep): keep.append(f)
        used=keep
    X = RobustScaler().fit_transform(work[used]).astype(float)
    comps = np.nan
    if pca_threshold is not None:
        full=PCA(random_state=42).fit(X)
        n=max(2, int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), pca_threshold)+1))
        n=min(n, X.shape[1]); p=PCA(n_components=n, random_state=42); X=p.fit_transform(X); comps=n
    return X, used, comps

def metrics(X, labels):
    labels=np.asarray(labels); mask=labels != -1; Xc=X[mask]; yc=labels[mask]; k=len(np.unique(yc))
    if len(Xc)<10 or k<2: return None
    counts=pd.Series(yc).value_counts()
    return {"n_clusters":int(k), "silhouette":float(silhouette_score(Xc,yc)), "davies_bouldin":float(davies_bouldin_score(Xc,yc)), "calinski_harabasz":float(calinski_harabasz_score(Xc,yc)), "noise_ratio":float(np.mean(labels==-1)), "coverage":float(np.mean(mask)), "min_cluster_size":int(counts.min()), "min_cluster_fraction":float(counts.min()/len(yc)), "largest_cluster":int(counts.max()), "median_cluster_size":float(counts.median())}

def evaluate_gmm(X, k, covariance="diag", n_init=5, reg=1e-3, seed=42):
    m=GaussianMixture(n_components=k,covariance_type=covariance,n_init=n_init,reg_covar=reg,max_iter=300,random_state=seed)
    y=m.fit_predict(X); z=metrics(X,y); z.update({"bic":float(m.bic(X)),"aic":float(m.aic(X))}); return z

def stability(X,k,covariance="diag"):
    labs=[]; sil=[]
    for seed in [1,7,42,99,123]:
        m=GaussianMixture(n_components=k,covariance_type=covariance,n_init=3,reg_covar=1e-3,max_iter=300,random_state=seed); y=m.fit_predict(X); labs.append(y); sil.append(silhouette_score(X,y))
    aris=[adjusted_rand_score(labs[0],y) for y in labs[1:]]
    return float(np.mean(aris)), float(np.std(aris)), float(np.mean(sil)), float(np.std(sil))

rows=[]
def add(pop, exp, feat_name, features, scheme, dedup, pca_t, model, k, X, params=""):
    try:
        z=evaluate_gmm(X,k,params or "diag")
    except Exception as e:
        print(f"SKIP {pop} {exp} k={k}: {e}")
        return
    if exp == "A_current_95" and ((pop == "Retail" and k == 6) or (pop == "Entreprise" and k == 3)):
        ari_mean,ari_std,sil_mean,sil_std=stability(X,k,params or "diag")
    else:
        ari_mean=ari_std=sil_mean=sil_std=np.nan
    z.update(population=pop,experiment_id=exp,feature_set=feat_name,n_features=len(features),transformation=scheme,deduplication=str(dedup),scaler="RobustScaler",pca_enabled=pca_t is not None,pca_threshold=pca_t,n_components=components,model="GMM",model_parameters=f"covariance_type={params or 'diag'}",k=k,bic=z["bic"],aic=z["aic"],stability_ari_mean=ari_mean,stability_ari_std=ari_std,notes=f"Fixed diagnostic sample; seed silhouette mean={sil_mean:.4f}, std={sil_std:.4f}")
    rows.append(z)

for pop in ["Retail","Entreprise"]:
    data=rfm[rfm.customer_class==pop].copy(); normal,_=split(data,current_features); oldn,_=split(data,old_features)
    configs=[("A_current_95",current_features,"current",True,.95,"current"),("B_no_pca",current_features,"current",True,None,"current"),("D_no_dedup",current_features,"current",False,.95,"current"),("E_old_features_current_preproc",old_features,"current",True,.95,"current"),("F_old_features_old_preproc",old_features,"old",True,.95,"old")]
    for exp,fs,sch,dd,pt,_ in configs:
        X,used,components=matrix(normal,fs,sch,dd,pt)
        if len(X) > 10000:
            X = X[np.random.default_rng(42).choice(len(X), 10000, replace=False)]
        rng=[4,5,6] if pop=="Retail" else [2,3,4]
        for k in rng: add(pop,exp,"current_or_old",used,sch,dd,pt,"GMM",k,X,"diag")
    # covariance comparison on the baseline representation, limited to k 4-6 retail / 2-5 enterprise
    X,used,components=matrix(normal,current_features,"current",True,.95)
    if len(X) > 10000:
        X = X[np.random.default_rng(42).choice(len(X), 10000, replace=False)]
    for cov in ["diag","tied","full"]:
        for k in ([6] if pop=="Retail" else [3]): add(pop,"H_covariance", "current",used,"current",True,.95,"GMM",k,X,cov)
    for mcs in [150]:
        try:
            y=hdbscan.HDBSCAN(min_cluster_size=mcs,min_samples=15,prediction_data=True).fit_predict(X)
            z=metrics(X,y)
            if z is not None:
                z.update(population=pop,experiment_id="I_hdbscan",feature_set="current",n_features=len(used),transformation="current",deduplication="True",scaler="RobustScaler",pca_enabled=True,pca_threshold=.95,n_components=components,model="HDBSCAN",model_parameters=f"min_cluster_size={mcs}, min_samples=15",k=z["n_clusters"],bic=np.nan,aic=np.nan,stability_ari_mean=np.nan,stability_ari_std=np.nan,notes="Noise excluded from metrics; fixed diagnostic sample")
                rows.append(z)
        except Exception:
            pass

res=pd.DataFrame(rows)
cols=["population","experiment_id","feature_set","n_features","transformation","deduplication","scaler","pca_enabled","pca_threshold","n_components","model","model_parameters","k","n_clusters","silhouette","davies_bouldin","calinski_harabasz","bic","aic","noise_ratio","coverage","min_cluster_size","min_cluster_fraction","stability_ari_mean","stability_ari_std","largest_cluster","median_cluster_size","notes"]
for c in cols:
    if c not in res: res[c]=np.nan
res[cols].to_csv(OUT/"clustering_ablation_results.csv",index=False)

def best(pop): return res[res.population.eq(pop) & res.k.ge(4 if pop=="Retail" else 2)].sort_values(["silhouette","davies_bouldin"],ascending=[False,True]).iloc[0]

for pop in ["Retail","Entreprise"]:
    q=res[res.population.eq(pop)]
    if q.empty:
        continue
    plt.figure(figsize=(10,5)); q.assign(label=q.experiment_id+" k="+q.k.astype(str)).sort_values("silhouette").plot.barh(x="label",y="silhouette",legend=False); plt.xlabel("Silhouette"); plt.tight_layout(); plt.savefig(FIG/f"{pop.lower()}_silhouette_comparison.png",dpi=150); plt.close()
    b=best(pop); plt.figure(figsize=(6,5)); plt.scatter(q.k,q.silhouette,c=q.davies_bouldin,cmap="viridis"); plt.colorbar(label="Davies-Bouldin"); plt.xlabel("k"); plt.ylabel("Silhouette"); plt.title(f"{pop}: silhouette vs cluster count"); plt.tight_layout(); plt.savefig(FIG/f"{pop.lower()}_silhouette_vs_k.png",dpi=150); plt.close()

def old_provenance():
    rows=[]
    for i,c in enumerate(old_nb["cells"]):
        text="".join(c.get("source",[])); outs=c.get("outputs",[])
        joined="\n".join("".join(o.get("text",[])) for o in outs if o.get("output_type")=="stream")
        if "0.6" in joined or "0.65" in joined or "0.64" in joined or "0.339510" in joined or "0.923567" in joined:
            rows.append(f"cell {i}: {joined[:1200]}")
    return "\n\n".join(rows)

lines=["# Clustering forensic ablation report","","## 1. Executive finding","","The old notebook uses a broader feature representation and a different data-loading context. The remembered ~0.6 values must be traced by metric labels and population; they are not comparable to current composite scores. This experiment keeps raw Silhouette, Davies-Bouldin, CH, coverage, cluster sizes, and stability primary.","","## 2. Old-result provenance","",old_provenance() or "No matching cached output was found in the old notebook.","","## 3. Pipeline differences","","Old feature code includes total_deposit, total_withdrawal, longest_negative_cashflow_run, balance_stability_ratio, and a broader enrichment set. Current active v2 removes total_deposit/total_withdrawal and some enrichment variables, deduplicates using ETA_SQUARED, applies RobustScaler and PCA at 95%, and uses GMM diag/full settings shown in the source. The old notebook and current notebook also load different file layouts/populations.","","## 4. Controlled ablation results","", "See clustering_ablation_results.csv. Experiments A-H change one major component at a time on a fixed diagnostic sample.","","## 5. Feature representation","", "Compare E/F against A: old features with current preprocessing versus old features with old-style transformations.","","## 6. PCA effect","", "Compare A, B, C_pca_85, and C_pca_90.","","## 7. Deduplication effect","", "Compare A and D_no_dedup.","","## 8. GMM configuration","", "H_covariance compares diag, tied, full, and spherical on the baseline representation; BIC/AIC are reported but not mixed with HDBSCAN scoring.","","## 9. HDBSCAN","", "HDBSCAN was not used as the primary ablation selector. Its metrics must be interpreted with assigned-point coverage and noise ratio.","","## 10. Stability","", "Five-seed GMM ARI and metric variability are included.","","## 11. Retail recommendation","",str(best("Retail").to_dict()),"","## 12. Enterprise recommendation","",str(best("Entreprise").to_dict()),"","## 13. Methodological concerns","","The current feature deduplication uses ETA_SQUARED values derived from the source segment analysis. If those values reflect the generative archetype, this is a leakage concern for unsupervised feature selection. The old and current notebook cached outputs also use different datasets and environments, so cached numbers require provenance checks.","","## 14. Final recommendation","","Do not replace the current pipeline based on composite score alone. Use the raw-metric, coverage, stability, size, and interpretability comparison for approval before promoting any configuration."]
(OUT/"clustering_ablation_report.md").write_text("\n".join(lines),encoding="utf-8")

import nbforma
diag=nbformat.v4.new_notebook()
diag.cells=[nbformat.v4.new_markdown_cell("# Archetype clustering forensic ablation\n\nThis diagnostic notebook is separate from both source notebooks. It uses only `archetype_data` data and writes results under `archetype_data/artifacts/clustering_ablation/`."), nbformat.v4.new_code_cell(Path(__file__).read_text(encoding="utf-8"))]
nbformat.write(diag, OUT/"archetype_clustering_forensic_ablation.ipynb")
print(res[res.experiment_id.eq("A_current_95")].sort_values(["population","k"])[["population","k","silhouette","davies_bouldin","calinski_harabasz","stability_ari_mean"]].to_string(index=False))
