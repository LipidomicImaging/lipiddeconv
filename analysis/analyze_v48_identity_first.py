#!/usr/bin/env python3
"""Identity-first analysis using frozen v48 truth and existing certificates only."""
from __future__ import annotations
import argparse, base64, io, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

FEATURES={
    "rho_zero":"higher",
    "necessity_signal":"higher",
    "profile_relative_width":"lower",
    "global_fragment_cone_residual":"higher",
}
COVERAGES=[.05,.10,.20,.40,.60,.80,1.0]

def auc_identity(y, trust):
    y=np.asarray(y,int); s=np.asarray(trust,float); m=np.isfinite(s); y=y[m]; s=s[m]
    n1=int(y.sum()); n0=len(y)-n1
    if not n1 or not n0:return np.nan
    return float((rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def quant(x,q): return float(pd.Series(x).quantile(q))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="v48_pilot_48"); ap.add_argument("--output",default=None); z=ap.parse_args()
    root=Path(z.input); out=Path(z.output or root/"identity_first"); out.mkdir(parents=True,exist_ok=True)
    man=pd.read_csv(root/"v48_pilot_manifest.csv")
    truth=pd.read_csv(root/"v48_pilot_lipid_observations_partial.csv")
    cert=pd.read_csv(root/"v48_pilot_lipid_observations.csv")
    meta=man[["case_id","K","residual_ratio","selection_mode","replicate"]]
    truth=truth.merge(meta,on="case_id",how="left",validate="many_to_one")
    rep0=meta[meta.replicate==0].copy(); ids=set(rep0.case_id.astype(int))
    if len(ids)!=16: raise RuntimeError(f"Expected 16 replicate=0 cases, got {len(ids)}")
    have=set(cert.case_id.astype(int)); missing=sorted(ids-have)
    if missing: raise RuntimeError(f"Existing certificate is missing replicate=0 cases: {missing}")
    cert16=cert[cert.case_id.isin(ids)].merge(meta,on="case_id",how="left",validate="many_to_one")
    if len(cert16)!=16*391 or cert16.duplicated(["case_id","candidate_index"]).any(): raise RuntimeError("replicate=0 certificate alignment failure")

    # PART A: all 48 cases, identity endpoints from existing X_true/X_hat.
    cases=[]
    for cid,g in truth.groupby("case_id",sort=True):
        K=int(g.K.iloc[0]); total=float(g.X_hat.sum()); fm=float(g.loc[g.X_true==0,"X_hat"].sum()/total) if total>0 else np.nan
        tm=float(g.loc[g.X_true>0,"X_hat"].sum()/total) if total>0 else np.nan
        top=set(g.nlargest(K,"X_hat").candidate_index.astype(int)); true=set(g.loc[g.X_true>0,"candidate_index"].astype(int)); recall=len(top&true)/K
        false_total=float(g.loc[g.X_true==0,"X_hat"].sum()); top10=float(g[g.X_true==0].nlargest(10,"X_hat").X_hat.sum())
        cases.append(dict(row_type="case",case_id=int(cid),K=K,residual_ratio=float(g.residual_ratio.iloc[0]),selection_mode=g.selection_mode.iloc[0],replicate=int(g.replicate.iloc[0]),false_allocation_mass=fm,true_mass_fraction=tm,consistency_abs_error=abs((1-fm)-tm),topK_identity_recall=recall,top10_false_mass_coverage=top10/false_total if false_total>0 else np.nan,reconstruction_relative_residual=float(g.reconstruction_relative_residual.iloc[0])))
    case_df=pd.DataFrame(cases)
    summaries=[]
    for axis in ["K","residual_ratio","selection_mode"]:
        for val,g in case_df.groupby(axis,sort=True):
            summaries.append(dict(row_type=f"summary_{axis}",group_value=val,n_cases=len(g),false_allocation_median=g.false_allocation_mass.median(),false_allocation_p25=quant(g.false_allocation_mass,.25),false_allocation_p75=quant(g.false_allocation_mass,.75),topK_recall_median=g.topK_identity_recall.median(),topK_recall_p25=quant(g.topK_identity_recall,.25),topK_recall_p75=quant(g.topK_identity_recall,.75),true_mass_consistency_max=g.consistency_abs_error.max()))
    for keys,g in case_df.groupby(["K","residual_ratio","selection_mode"],sort=True):
        summaries.append(dict(row_type="summary_scenario",K=keys[0],residual_ratio=keys[1],selection_mode=keys[2],n_cases=len(g),false_allocation_median=g.false_allocation_mass.median(),false_allocation_p25=quant(g.false_allocation_mass,.25),false_allocation_p75=quant(g.false_allocation_mass,.75),topK_recall_median=g.topK_identity_recall.median(),topK_recall_p25=quant(g.topK_identity_recall,.25),topK_recall_p75=quant(g.topK_identity_recall,.75),true_mass_consistency_max=g.consistency_abs_error.max()))
    truth_summary=pd.concat([case_df,pd.DataFrame(summaries)],ignore_index=True,sort=False)
    truth_summary.to_csv(out/"v48_identity_truth_summary.csv",index=False)

    # PART D/F: every true-active plus top-10 false allocations in each of 16 cases.
    keep=[]
    for cid,g in cert16.groupby("case_id",sort=True):
        keep.extend(g[g.X_true>0].index.tolist()); keep.extend(g[g.X_true==0].nlargest(10,"X_hat").index.tolist())
    rec=cert16.loc[sorted(set(keep))].copy(); rec["truth_identity"]=(rec.X_true>0).astype(int); rec["estimated_abundance"]=rec.X_hat
    rec=rec.rename(columns={"competition_group_id":"competition_group"})
    cols=["case_id","K","residual_ratio","selection_mode","candidate_index","candidate_id","lipid_name","X_true","X_hat","truth_identity","estimated_abundance","rho_zero","necessity_signal","profile_lower","profile_upper","profile_relative_width","global_fragment_cone_residual","competition_group"]
    rec[cols].to_parquet(out/"v48_identity_certificate_records.parquet",index=False)

    # PART G/H: fixed-direction univariate identity evidence.
    frows=[]; curves={}
    for f,direction in FEATURES.items():
        if f not in rec or rec[f].notna().sum()<3: continue
        m=rec[np.isfinite(rec[f])].copy(); m["trust_score"]=m[f] if direction=="higher" else -m[f]
        auc=auc_identity(m.truth_identity,m.trust_score)
        frows.append(dict(row_type="distribution",feature=f,direction=direction,n=len(m),n_true=int(m.truth_identity.sum()),n_false=int((1-m.truth_identity).sum()),identity_AUROC=auc,true_median=m.loc[m.truth_identity==1,f].median(),false_median=m.loc[m.truth_identity==0,f].median()))
        m=m.sort_values(["trust_score","case_id","candidate_index"],ascending=[False,True,True],kind="mergesort").reset_index(drop=True)
        m["trust_quintile"]=(np.floor(np.arange(len(m))*5/len(m))+1).astype(int)
        for q,g in m.groupby("trust_quintile"):
            mass=float(g.X_hat.sum()); false=float(g.loc[g.truth_identity==0,"X_hat"].sum())
            frows.append(dict(row_type="quintile",feature=f,direction=direction,trust_quintile=int(q),n=len(g),fraction_truth_identity_1=g.truth_identity.mean(),fraction_truth_identity_0=1-g.truth_identity.mean(),mass_weighted_false_fraction=false/mass if mass>0 else np.nan))
        if f in ["rho_zero","profile_relative_width"]:
            curves[f]=[]
            for cov in COVERAGES:
                g=m.iloc[:max(1,int(np.ceil(cov*len(m))))]; mass=float(g.X_hat.sum()); false=float(g.loc[g.truth_identity==0,"X_hat"].sum())
                row=dict(row_type="risk_coverage",feature=f,direction=direction,coverage=cov,n=len(g),identity_precision=g.truth_identity.mean(),mass_weighted_false_fraction=false/mass if mass>0 else np.nan)
                frows.append(row); curves[f].append(row)
    fs=pd.DataFrame(frows); fs.to_csv(out/"v48_identity_certificate_summary.csv",index=False)

    # PART I: use all false mass in the same 16 cases; group IDs are frozen lookups.
    true_groups={cid:set(g.loc[(g.X_true>0)&g.competition_group.notna(),"competition_group"]) for cid,g in rec.groupby("case_id")}
    gres=[]
    for cid,g in cert16.groupby("case_id",sort=True):
        fg=g[(g.X_true==0)&(g.X_hat>0)].copy(); tg=true_groups.get(cid,set())
        within=fg.competition_group_id.notna() & fg.competition_group_id.isin(tg)
        allmass=float(fg.X_hat.sum()); win=float(fg.loc[within,"X_hat"].sum()); outside=allmass-win
        trueg=g[g.X_true>0].copy(); trueg["member_error"]=(trueg.X_hat-trueg.X_true).abs()/trueg.X_true
        groups=g[g.competition_group_id.notna()].groupby("competition_group_id")[["X_true","X_hat"]].sum(); groups=groups[groups.X_true>0]; ge=(groups.X_hat-groups.X_true).abs()/groups.X_true if len(groups) else pd.Series(dtype=float)
        gres.append(dict(row_type="case",case_id=int(cid),K=int(g.K.iloc[0]),residual_ratio=float(g.residual_ratio.iloc[0]),selection_mode=g.selection_mode.iloc[0],total_false_mass=allmass,within_group_false_mass=win,outside_group_false_mass=outside,within_group_false_mass_fraction=win/allmass if allmass>0 else np.nan,outside_group_false_mass_fraction=outside/allmass if allmass>0 else np.nan,member_median_error=trueg.member_error.median(),group_total_median_error=ge.median() if len(ge) else np.nan))
    gd=pd.DataFrame(gres); total=gd.total_false_mass.sum(); aggregate=dict(row_type="overall",n_cases=len(gd),total_false_mass=total,within_group_false_mass=gd.within_group_false_mass.sum(),outside_group_false_mass=gd.outside_group_false_mass.sum(),within_group_false_mass_fraction=gd.within_group_false_mass.sum()/total,outside_group_false_mass_fraction=gd.outside_group_false_mass.sum()/total,member_median_error=gd.member_median_error.median(),group_total_median_error=gd.group_total_median_error.median())
    groupout=pd.concat([gd,pd.DataFrame([aggregate])],ignore_index=True); groupout.to_csv(out/"v48_identity_group_rescue.csv",index=False)

    byK=pd.DataFrame(summaries); byK=byK[byK.row_type=="summary_K"]
    dist=fs[fs.row_type=="distribution"].set_index("feature")
    report={"status":"IDENTITY_FIRST_PILOT_COMPLETE","scope":{"truth_cases":48,"identity_certificate_cases":16,"false_candidates_per_case":10,"new_certificate_calculations":0},"by_K":byK.to_dict("records"),"rho_zero_AUROC":float(dist.loc["rho_zero","identity_AUROC"]),"profile_width_AUROC":float(dist.loc["profile_relative_width","identity_AUROC"]),"risk_coverage":curves,"group_rescue":aggregate,"notes":["Identity AUROC directions were fixed before inspection.","Risk-coverage population contains all true-active records and top-10 false allocations per replicate=0 case.","No threshold or classifier was fitted."]}
    (out/"v48_identity_first_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,default=lambda o:o.item() if hasattr(o,"item") else str(o)),encoding="utf-8")

    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,f in zip(axs,["rho_zero","profile_relative_width"]):
        q=pd.DataFrame(curves[f]); ax.plot(100*q.coverage,100*q.mass_weighted_false_fraction,marker="o",label="false allocation mass")
        ax.plot(100*q.coverage,100*(1-q.identity_precision),marker="s",label="false identity count")
        ax.set(title=f,xlabel="coverage retained (%)",ylabel="error (%)",ylim=(0,100)); ax.grid(alpha=.25); ax.legend(fontsize=8)
    b=io.BytesIO(); fig.tight_layout(); fig.savefig(b,format="png",dpi=140); plt.close(fig); uri="data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
    rc_rho=pd.DataFrame(curves["rho_zero"]); rc_pw=pd.DataFrame(curves["profile_relative_width"])
    html="""<!doctype html><meta charset='utf-8'><title>v48 identity-first</title><style>body{font-family:Arial;margin:32px;max-width:1200px}table{border-collapse:collapse}th,td{padding:5px 8px;border:1px solid #ddd}img{max-width:100%}.lead{font-size:20px}</style>"""
    html+=f"<h1>v48 identity-first pilot</h1><p class='lead'>PRIMARY: identity correctness / false allocation; SECONDARY: group rescue; TERTIARY: abundance error.</p><ol><li>随 K 增大：见下表 false-allocation 与 Top-K recall。</li><li>Top-K identity recall：见下表。</li><li>rho_zero 固定方向 identity AUROC={report['rho_zero_AUROC']:.3f}。</li><li>profile width 固定方向 identity AUROC={report['profile_width_AUROC']:.3f}。</li><li>最可信 5/10/20% 的 false mass：rho_zero={rc_rho.head(3).mass_weighted_false_fraction.map(lambda x:f'{x:.2%}').tolist()}；profile width={rc_pw.head(3).mass_weighted_false_fraction.map(lambda x:f'{x:.2%}').tolist()}。</li><li>false mass：组内交换={aggregate['within_group_false_mass_fraction']:.2%}；组外错误={aggregate['outside_group_false_mass_fraction']:.2%}。</li></ol><h2>Identity truth by K</h2>{byK.to_html(index=False)}<h2>Certificate distributions</h2>{dist.reset_index().to_html(index=False)}<h2>Risk–coverage（不选择 cutoff）</h2><img src='{uri}'><h2>Group rescue</h2>{pd.DataFrame([aggregate]).to_html(index=False)}<h2>Tertiary quantitative context</h2><p>保留既有 relative-error 结果，但不作为本报告首要终点。本分析没有训练 classifier、没有搜索 threshold，也没有启动 v49。</p>"
    (out/"v48_identity_first_report.html").write_text(html,encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o)))

if __name__=="__main__": main()
