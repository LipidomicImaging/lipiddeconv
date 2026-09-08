#!/usr/bin/env python3
"""Read-only 90%-false-mass validation for the frozen v48 identity pilot."""
from __future__ import annotations
import argparse, base64, io, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

FEATURES={"rho_zero":"higher","necessity_signal":"higher","profile_relative_width":"lower","global_fragment_cone_residual":"higher"}
COVERAGES=(.05,.10,.20,.40,.60,.80,1.0)

def auc(y, score):
    y=np.asarray(y,dtype=int); s=np.asarray(score,dtype=float); m=np.isfinite(s); y,s=y[m],s[m]
    n1=int(y.sum()); n0=len(y)-n1
    if n1==0 or n0==0:return np.nan
    return float((rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def triplet(s):
    s=pd.to_numeric(s,errors="coerce").dropna()
    return (float(s.quantile(.25)),float(s.median()),float(s.quantile(.75))) if len(s) else (None,None,None)

def feature_stats(records, strata):
    out=[]; curves={}
    groups=[("overall","all",records)] + [(c,str(v),g) for c in strata for v,g in records.groupby(c,sort=True)]
    for f,direction in FEATURES.items():
        for typ,val,g0 in groups:
            g=g0[np.isfinite(g0[f])].copy()
            if not len(g):continue
            g["trust"]=g[f] if direction=="higher" else -g[f]
            g=g.sort_values(["trust","case_id","candidate_index"],ascending=[False,True,True],kind="mergesort").reset_index(drop=True)
            if typ=="overall":
                t25,t50,t75=triplet(g.loc[g.truth_identity==1,f]); z25,z50,z75=triplet(g.loc[g.truth_identity==0,f])
                out.append(dict(row_type="distribution",feature=f,direction=direction,stratum_type=typ,stratum_value=val,n=len(g),n_true=int(g.truth_identity.sum()),n_false=int((g.truth_identity==0).sum()),identity_AUROC=auc(g.truth_identity,g.trust),true_p25=t25,true_median=t50,true_p75=t75,false_p25=z25,false_median=z50,false_p75=z75,interpretation="contains a near-zero-reference denominator effect" if f=="profile_relative_width" else ""))
            points=(COVERAGES if typ=="overall" else (.10,.20))
            for cov in points:
                z=g.iloc[:max(1,int(np.ceil(cov*len(g))))]; mass=float(z.X_hat.sum()); fm=float(z.loc[z.truth_identity==0,"X_hat"].sum())
                row=dict(row_type="risk_coverage",feature=f,direction=direction,stratum_type=typ,stratum_value=val,coverage=cov,n=len(z),identity_precision=float(z.truth_identity.mean()),mass_weighted_false_fraction=fm/mass if mass>0 else np.nan)
                out.append(row)
                if typ=="overall" and f in ("rho_zero","profile_relative_width"): curves.setdefault(f,[]).append(row)
    return pd.DataFrame(out),curves

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="v48_pilot_48"); ap.add_argument("--output",default=None); a=ap.parse_args()
    root=Path(a.input); out=Path(a.output or root/"identity_mass90"); out.mkdir(parents=True,exist_ok=True)
    p=root/"v48_pilot_lipid_observations.csv"; man=pd.read_csv(root/"v48_pilot_manifest.csv"); d=pd.read_csv(p)
    if len(d)!=48*391 or d.case_id.nunique()!=48 or d.duplicated(["case_id","candidate_index"]).any(): raise RuntimeError("Full cached certificate table is not 48x391 aligned")
    meta=man[["case_id","K","residual_ratio","selection_mode","replicate"]]
    duplicate=[c for c in meta.columns if c!="case_id" and c in d.columns]; d=d.drop(columns=duplicate).merge(meta,on="case_id",how="left",validate="many_to_one")
    selected=[]; crows=[]
    for cid,g in d.groupby("case_id",sort=True):
        false=g[(g.X_true==0)&(g.X_hat>0)].sort_values("X_hat",ascending=False,kind="mergesort").copy(); total=float(false.X_hat.sum())
        if total>0:
            false["cum_fraction"]=false.X_hat.cumsum()/total; n=int(np.argmax(false.cum_fraction.to_numpy()>=.90)+1); f90=false.iloc[:n]
            cov=float(f90.X_hat.sum()/total)
        else: n=0; f90=false.iloc[:0]; cov=np.nan
        true=g[g.X_true>0]
        selected.append(true); selected.append(f90)
        crows.append(dict(case_id=int(cid),K=int(g.K.iloc[0]),residual_ratio=float(g.residual_ratio.iloc[0]),selection_mode=g.selection_mode.iloc[0],replicate=int(g.replicate.iloc[0]),n_false_total=int(len(false)),n_false_mass90=n,false_mass90_coverage=cov,total_false_mass=total))
    records=pd.concat(selected,ignore_index=True); records["truth_identity"]=(records.X_true>0).astype(int); records["estimated_abundance"]=records.X_hat; records=records.rename(columns={"competition_group_id":"competition_group"})
    cols=["case_id","candidate_id","candidate_index","K","residual_ratio","selection_mode","replicate","X_true","X_hat","truth_identity","estimated_abundance","rho_zero","necessity_signal","profile_lower","profile_upper","profile_relative_width","global_fragment_cone_residual","competition_group"]
    records[cols].to_parquet(out/"v48_identity_mass90_records.parquet",index=False)
    cs=pd.DataFrame(crows); cs.to_csv(out/"v48_identity_mass90_case_summary.csv",index=False)
    if not ((cs.false_mass90_coverage.dropna()>=.90).all()): raise RuntimeError("90% selection coverage failure")
    fs,curves=feature_stats(records,["K","residual_ratio","selection_mode"]); fs.to_csv(out/"v48_identity_mass90_feature_summary.csv",index=False)
    # Group rescue only for the selected mass90 false set; frozen group IDs are cached columns.
    gres=[]
    for cid,g in d.groupby("case_id",sort=True):
        rec=records[records.case_id==cid]; false=rec[rec.truth_identity==0]; true_groups=set(g.loc[(g.X_true>0)&g.competition_group_id.notna(),"competition_group_id"])
        inside=false.competition_group.notna() & false.competition_group.isin(true_groups)
        total=float(false.X_hat.sum()); win=float(false.loc[inside,"X_hat"].sum())
        gres.append(dict(row_type="case",case_id=int(cid),K=int(g.K.iloc[0]),residual_ratio=float(g.residual_ratio.iloc[0]),selection_mode=g.selection_mode.iloc[0],selected_false_mass=total,within_group_false_mass=win,outside_group_false_mass=total-win,within_group_false_mass_fraction=win/total if total else np.nan,outside_group_false_mass_fraction=(total-win)/total if total else np.nan))
    gr=pd.DataFrame(gres); total=float(gr.selected_false_mass.sum()); old=json.loads((root/"identity_first"/"v48_identity_first_report.json").read_text())
    oldg=old["group_rescue"]
    overall=dict(row_type="overall",n_cases=48,selected_false_mass=total,within_group_false_mass=float(gr.within_group_false_mass.sum()),outside_group_false_mass=float(gr.outside_group_false_mass.sum()),within_group_false_mass_fraction=float(gr.within_group_false_mass.sum()/total),outside_group_false_mass_fraction=float(gr.outside_group_false_mass.sum()/total),old_top10_within_group_fraction=oldg["within_group_false_mass_fraction"],old_top10_outside_group_fraction=oldg["outside_group_false_mass_fraction"])
    gr=pd.concat([gr,pd.DataFrame([overall])],ignore_index=True); gr.to_csv(out/"v48_identity_mass90_group_summary.csv",index=False)
    oldrho=float(old["rho_zero_AUROC"]); oldpw=float(old["profile_width_AUROC"]); oldcurves=old["risk_coverage"]
    sanity=json.loads((root/"identity_sanity"/"v48_identity_sanity_check.json").read_text())
    old_coverage=float(sanity["top10_false_mass_coverage"]["overall"]["median"])
    def cv(curveset,cov): return next(x["mass_weighted_false_fraction"] for x in curveset if abs(x["coverage"]-cov)<1e-12)
    dist=fs[(fs.row_type=="distribution")&(fs.stratum_type=="overall")].set_index("feature")
    newrho=float(dist.loc["rho_zero","identity_AUROC"]); newpw=float(dist.loc["profile_relative_width","identity_AUROC"])
    compare=pd.DataFrame([
        {"metric":"false-mass coverage (case median)","top10_false_pilot":old_coverage,"mass90_validation":float(cs.false_mass90_coverage.median()),"note":"fraction of each case's false allocation mass"},
        {"metric":"rho_zero AUROC","top10_false_pilot":oldrho,"mass90_validation":newrho,"note":"fixed higher-is-trustworthy direction"},
        {"metric":"rho_zero top10% false mass","top10_false_pilot":cv(oldcurves["rho_zero"],.10),"mass90_validation":cv(curves["rho_zero"],.10),"note":"enriched identity datasets"},
        {"metric":"rho_zero top20% false mass","top10_false_pilot":cv(oldcurves["rho_zero"],.20),"mass90_validation":cv(curves["rho_zero"],.20),"note":"enriched identity datasets"},
        {"metric":"profile-width AUROC","top10_false_pilot":oldpw,"mass90_validation":newpw,"note":"contains near-zero-reference denominator effect"},
        {"metric":"within-group false fraction","top10_false_pilot":oldg["within_group_false_mass_fraction"],"mass90_validation":overall["within_group_false_mass_fraction"],"note":"frozen d_frag=0.02 groups"},
    ])
    rho_strata=fs[(fs.row_type=="risk_coverage")&(fs.feature=="rho_zero")&(fs.coverage.isin([.10,.20]))]
    worst=float(rho_strata[rho_strata.stratum_type!="overall"].mass_weighted_false_fraction.max())
    pooled20=float(cv(curves["rho_zero"],.20))
    status="ROBUST_IDENTITY_SIGNAL" if pooled20<=.01 and worst<=.05 else ("CONDITIONAL_IDENTITY_SIGNAL" if pooled20<=.05 else "TOP10_SELECTION_BIAS")
    report={"status":status,"scope":{"cached_certificate_rows":len(d),"cases":48,"candidates_per_case":391,"no_new_certificate_calculations":True},"mass90_selection":{"n_false_mass90_p25":float(cs.n_false_mass90.quantile(.25)),"median":float(cs.n_false_mass90.median()),"p75":float(cs.n_false_mass90.quantile(.75)),"max":int(cs.n_false_mass90.max()),"coverage_min":float(cs.false_mass90_coverage.min()),"coverage_median":float(cs.false_mass90_coverage.median()),"coverage_max":float(cs.false_mass90_coverage.max())},"rho_zero_AUROC":newrho,"profile_relative_width_AUROC":newpw,"rho_zero_risk_coverage":curves["rho_zero"],"rho_zero_stratified_top10_top20":rho_strata.to_dict("records"),"group_rescue_mass90":overall,"top10_vs_mass90":compare.to_dict("records"),"interpretation":"profile_relative_width contains a near-zero-reference denominator effect"}
    (out/"v48_identity_mass90_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o)),encoding="utf-8")
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,f in zip(axs,["rho_zero","profile_relative_width"]):
        q=pd.DataFrame(curves[f]); ax.plot(100*q.coverage,100*q.mass_weighted_false_fraction,marker="o"); ax.set(title=f,xlabel="coverage retained (%)",ylabel="false allocation mass (%)",ylim=(0,100)); ax.grid(alpha=.25)
    b=io.BytesIO(); fig.tight_layout(); fig.savefig(b,format="png",dpi=140); plt.close(fig); uri="data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
    h=lambda x:x.to_html(index=False,float_format=lambda v:f"{v:.4g}")
    html=f"""<!doctype html><meta charset='utf-8'><title>v48 mass90 identity</title><style>body{{font-family:Arial;margin:32px;max-width:1300px}}table{{border-collapse:collapse}}th,td{{padding:5px 8px;border:1px solid #ddd}}img{{max-width:100%}}.status{{font-size:28px;font-weight:bold}}</style><h1>v48 identity mass-complete validation</h1><div class='status'>{status}</div><p>Pure cached-data analysis; no certificate, profile, ISTA, or simulation was recomputed.</p><ol><li>每个 case 覆盖至少90% false mass；median={cs.false_mass90_coverage.median():.2%}。</li><li>rho_zero AUROC={newrho:.3f}; profile width AUROC={newpw:.3f}（含 near-zero-reference denominator effect）。</li><li>rho_zero top10/top20 false mass={cv(curves['rho_zero'],.10):.2%}/{cv(curves['rho_zero'],.20):.2%}。</li><li>Mass90 false mass 组内={overall['within_group_false_mass_fraction']:.2%}，组外={overall['outside_group_false_mass_fraction']:.2%}。</li></ol><h2>90% selection</h2>{h(cs.describe())}<h2>Feature evidence</h2>{h(dist.reset_index())}<h2>Top10 versus mass90</h2>{h(compare)}<h2>Group rescue</h2>{h(pd.DataFrame([overall]))}<h2>Risk–coverage</h2><img src='{uri}'><h2>rho_zero strata</h2>{h(rho_strata)}"""
    (out/"v48_identity_mass90_report.html").write_text(html,encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o)))

if __name__=="__main__": main()
