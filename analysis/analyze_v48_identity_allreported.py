#!/usr/bin/env python3
"""Read-only all-reported-candidate identity validation for frozen v48 output."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

GATES=(1e-4,1e-3,1e-2)
COVERAGES=(.05,.10,.20,.40,.60,.80,1.)
FEATURES={"rho_zero":"higher","necessity_signal":"higher","profile_relative_width":"lower","global_fragment_cone_residual":"higher"}

def auc(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); m=np.isfinite(s); y,s=y[m],s[m]; n1=int(y.sum()); n0=len(y)-n1
    if not n1 or not n0:return np.nan
    return float((rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def risk(g, feature, covs, total_true_identities):
    d=g[np.isfinite(g[feature])].copy(); direction=FEATURES[feature]
    d["trust"]=d[feature] if direction=="higher" else -d[feature]
    d=d.sort_values(["trust","case_id","candidate_index"],ascending=[False,True,True],kind="mergesort").reset_index(drop=True)
    rows=[]
    for cov in covs:
        z=d.iloc[:max(1,int(np.ceil(cov*len(d))))]; mass=float(z.X_hat.sum()); false=float(z.loc[z.truth_identity==0,"X_hat"].sum())
        precision=float(z.truth_identity.mean())
        rows.append(dict(coverage=cov,n_reported=len(z),identity_precision=precision,false_identity_fraction=float(1-precision),mass_weighted_false_fraction=false/mass if mass else np.nan,true_identity_recall=float(z.truth_identity.sum()/total_true_identities) if total_true_identities else np.nan))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="v48_pilot_48"); ap.add_argument("--output",default=None); a=ap.parse_args()
    root=Path(a.input); out=Path(a.output or root/"identity_allreported"); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(root/"v48_pilot_lipid_observations.csv"); man=pd.read_csv(root/"v48_pilot_manifest.csv")
    if len(d)!=48*391 or d.case_id.nunique()!=48 or d.duplicated(["case_id","candidate_index"]).any(): raise RuntimeError("Expected full 48x391 cached certificate table")
    meta=man[["case_id","K","residual_ratio","selection_mode","replicate"]]; du=[c for c in meta if c!="case_id" and c in d]; d=d.drop(columns=du).merge(meta,on="case_id",how="left",validate="many_to_one")
    d["truth_identity"]=(d.X_true>0).astype(int)
    summary=[]; rc=[]; report_gates={}
    for gate in GATES:
        x=d[d.X_hat>gate].copy(); mass=float(x.X_hat.sum()); fm=float(x.loc[x.truth_identity==0,"X_hat"].sum()); total_true=int(d.truth_identity.sum())
        precision=float(x.truth_identity.mean())
        summary.append(dict(row_type="gate_baseline",gate=gate,n_reported=len(x),n_true_reported=int(x.truth_identity.sum()),n_false_reported=int((x.truth_identity==0).sum()),identity_precision=precision,false_identity_fraction=float(1-precision),mass_weighted_false_fraction=fm/mass if mass else np.nan,true_identity_recall=float(x.truth_identity.sum()/total_true)))
        feat={}
        for f,direction in FEATURES.items():
            z=x[np.isfinite(x[f])]
            trust=z[f] if direction=="higher" else -z[f]
            feat[f]={"identity_AUROC":auc(z.truth_identity,trust),"true_median":float(z.loc[z.truth_identity==1,f].median()) if (z.truth_identity==1).any() else None,"false_median":float(z.loc[z.truth_identity==0,f].median()) if (z.truth_identity==0).any() else None,"note":"contains a near-zero-reference denominator effect" if f=="profile_relative_width" else ("negative comparator" if f=="global_fragment_cone_residual" else "")}
            summary.append(dict(row_type="feature_AUROC",gate=gate,feature=f,direction=direction,n=len(z),**feat[f]))
        rr=risk(x,"rho_zero",COVERAGES,total_true)
        for q in rr: rc.append(dict(row_type="pooled_rho_zero",gate=gate,feature="rho_zero",stratum_type="overall",stratum_value="all",**q))
        # Necessity is reported in parallel but is not the requested primary curve.
        for q in risk(x,"necessity_signal",COVERAGES,total_true): rc.append(dict(row_type="pooled_necessity",gate=gate,feature="necessity_signal",stratum_type="overall",stratum_value="all",**q))
        for col in ("K","residual_ratio","selection_mode"):
            for val,g in x.groupby(col,sort=True):
                full_true=int(d.loc[d[col]==val,"truth_identity"].sum())
                for q in risk(g,"rho_zero",(.10,.20),full_true):
                    rc.append(dict(row_type="rho_zero_stratified",gate=gate,feature="rho_zero",stratum_type=col,stratum_value=val,**q))
        report_gates[str(gate)]={"baseline":summary[-5],"features":feat,"rho_zero_risk_coverage":rr}
    sdf=pd.DataFrame(summary); rdf=pd.DataFrame(rc); sdf.to_csv(out/"v48_identity_allreported_summary.csv",index=False); rdf.to_csv(out/"v48_identity_allreported_riskcoverage.csv",index=False)
    feature_summary=sdf[sdf.row_type=="feature_AUROC"].copy(); feature_summary.to_csv(out/"v48_identity_allreported_feature_summary.csv",index=False)
    report={"status":"ALL_REPORTED_CANDIDATE_IDENTITY_VALIDATION_COMPLETE","scope":{"cached_rows":len(d),"cases":48,"no_new_calculations":True,"gates":list(GATES)},"gates":report_gates,"notes":["rho_zero is primary with fixed higher-is-trustworthy direction.","necessity_signal is parallel evidence; profile_relative_width is secondary and carries a near-zero-reference denominator effect.","global_fragment_cone_residual is retained as a negative comparator.","No classifier, threshold selection, simulation, ISTA, profile, or certificate was run."]}
    (out/"v48_identity_allreported_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o)),encoding="utf-8")
    base=sdf[sdf.row_type=="gate_baseline"]; au=feature_summary; rho=rdf[(rdf.row_type=="pooled_rho_zero")&(rdf.coverage.isin([.05,.10,.20,.40,.60,.80,1.0]))]
    html="""<!doctype html><meta charset='utf-8'><title>v48 all reported identity</title><style>body{font-family:Arial;margin:32px;max-width:1300px}table{border-collapse:collapse}th,td{padding:5px 8px;border:1px solid #ddd}</style>"""
    html+="<h1>v48 all-reported-candidate identity validation</h1><p>Pure cached-data analysis; no solver or certificate computation occurred.</p>"
    html+=f"<h2>Reporting gates</h2>{base.to_html(index=False)}<h2>Feature AUROC (fixed direction)</h2>{au.to_html(index=False)}<h2>rho_zero fixed risk coverage</h2>{rho.to_html(index=False)}<h2>rho_zero top10/top20 strata</h2>{rdf[rdf.row_type=='rho_zero_stratified'].to_html(index=False)}<p>Profile relative width is secondary because it contains a near-zero-reference denominator effect. Global cone is a negative comparator.</p>"
    (out/"v48_identity_allreported_report.html").write_text(html,encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o)))

if __name__=="__main__": main()
