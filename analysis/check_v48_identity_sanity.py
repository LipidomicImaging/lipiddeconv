#!/usr/bin/env python3
"""Read-only sanity checks for the frozen v48 identity-first pilot."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES={"rho_zero":"higher","profile_relative_width":"lower"}
COVERAGES=(.10,.20)
EPS=1e-12

def stats(s):
    s=pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    return {"n":int(len(s)),"p25":float(s.quantile(.25)) if len(s) else None,"median":float(s.median()) if len(s) else None,"p75":float(s.quantile(.75)) if len(s) else None}

def risk_rows(df, strata):
    out=[]
    groups=[("overall","all",df)]
    for col in strata:
        groups += [(col,str(v),g) for v,g in df.groupby(col,sort=True)]
    for stype,sval,g0 in groups:
        for feature,direction in FEATURES.items():
            g=g0[np.isfinite(g0[feature])].copy()
            g["trust"]=g[feature] if direction=="higher" else -g[feature]
            g=g.sort_values(["trust","case_id","candidate_index"],ascending=[False,True,True],kind="mergesort")
            for cov in COVERAGES:
                z=g.iloc[:max(1,int(np.ceil(cov*len(g))))]; mass=float(z.X_hat.sum()); false=float(z.loc[z.truth_identity==0,"X_hat"].sum())
                out.append({"stratum_type":stype,"stratum_value":sval,"feature":feature,"coverage":cov,"n":int(len(z)),"identity_precision":float(z.truth_identity.mean()),"mass_weighted_false_fraction":false/mass if mass>0 else None})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="v48_pilot_48"); ap.add_argument("--output",default=None); a=ap.parse_args()
    root=Path(a.input); ident=root/"identity_first"/"v48_identity_certificate_records.parquet"; cached=root/"v48_pilot_lipid_observations.csv"; ts=root/"identity_first"/"v48_identity_truth_summary.csv"; out=Path(a.output or root/"identity_sanity"); out.mkdir(parents=True,exist_ok=True)
    r=pd.read_parquet(ident); c=pd.read_csv(cached); truth=pd.read_csv(ts)
    false=r[r.truth_identity==0].copy(); keys=["case_id","candidate_id"]
    fields=["rho_zero","necessity_signal","profile_lower","profile_upper","profile_relative_width"]
    src=c[keys+fields].copy(); merged=false[keys+fields].merge(src,on=keys,how="left",suffixes=("_identity","_cached"),validate="one_to_one",indicator=True)
    comparisons={}
    all_match=bool((merged._merge=="both").all())
    for f in fields:
        x=pd.to_numeric(merged[f+"_identity"],errors="coerce").to_numpy(); y=pd.to_numeric(merged[f+"_cached"],errors="coerce").to_numpy(); ok=np.isclose(x,y,rtol=1e-12,atol=1e-15,equal_nan=True)
        comparisons[f]={"matching_rows":int(ok.sum()),"mismatching_rows":int((~ok).sum())}; all_match &= bool(ok.all())
    provenance={"status":"PASS" if len(false)==160 and false.duplicated(keys).sum()==0 and all_match else "FAIL","identity_records_path":str(ident.resolve()),"cached_certificate_source_path":str(cached.resolve()),"source_row_count":int(len(c)),"identity_false_row_count":int(len(false)),"identity_false_unique_case_candidate_count":int(false.drop_duplicates(keys).shape[0]),"merge_key":keys,"matched_rows":int((merged._merge=="both").sum()),"field_sources":{f:str(cached.resolve()) for f in fields},"field_exact_checks":comparisons,"new_certificate_calculations":0}

    width=[]
    xref_candidates=[x for x in ["x_reference","x_NNLS","x_nnls","x_ref"] if x in r.columns]
    for label,g in [("TRUE",r[r.truth_identity==1]),("FALSE",false)]:
        w=g.profile_upper-g.profile_lower
        implied=(w/g.profile_relative_width.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
        row={"identity":label,"absolute_profile_width":stats(w),"profile_relative_width":stats(g.profile_relative_width),"estimated_abundance_X_hat":stats(g.X_hat),"fraction_X_hat_le_1e-12":float((g.X_hat<=EPS).mean()),"implied_relative_width_denominator":stats(implied)}
        if xref_candidates:
            xr=g[xref_candidates[0]]; row["x_reference_field"]=xref_candidates[0]; row["x_reference"]=stats(xr); row["fraction_x_reference_le_numerical_epsilon"]=float((xr<=EPS).mean())
        else:
            row["x_reference_field"]="MISSING"; row["x_reference"]="MISSING"; row["fraction_x_reference_le_numerical_epsilon"]="MISSING"
        width.append(row)

    risk=risk_rows(r,["K","residual_ratio","selection_mode"])
    case=truth[truth.row_type=="case"].copy(); pilot=case[case.replicate==0]
    coverage={"scope":"16 replicate=0 identity-pilot cases","overall":stats(pilot.top10_false_mass_coverage),"by_K":{str(int(k)):float(g.top10_false_mass_coverage.median()) for k,g in pilot.groupby("K")},"all_48_context_overall":stats(case.top10_false_mass_coverage)}
    report={"status":"PASS" if provenance["status"]=="PASS" else "STOP_CANDIDATE_MISMATCH","provenance":provenance,"relative_width_sanity":width,"risk_coverage_stratified":risk,"top10_false_mass_coverage":coverage,"notes":["No ISTA/profile/certificate was recomputed.","x_reference/x_NNLS is reported missing when absent; X_hat is shown separately and is not relabeled as a reference solution.","Risk coverage uses the existing 944-record enriched identity dataset."]}
    (out/"v48_identity_sanity_check.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o)),encoding="utf-8")
    provtab=pd.DataFrame([{"status":provenance["status"],"source_path":provenance["cached_certificate_source_path"],"source_rows":provenance["source_row_count"],"false_rows":provenance["identity_false_row_count"],"unique_case_candidate":provenance["identity_false_unique_case_candidate_count"],"merge_key":" + ".join(keys),"matched_rows":provenance["matched_rows"]}])
    wtab=pd.json_normalize(width)
    rtab=pd.DataFrame(risk)
    html="""<!doctype html><meta charset='utf-8'><title>v48 identity sanity</title><style>body{font-family:Arial;margin:32px;max-width:1300px}table{border-collapse:collapse}th,td{padding:5px 8px;border:1px solid #ddd}.pass{font-size:28px;font-weight:bold;color:#176b35}</style>"""
    html+=f"<h1>v48 identity-first minimal sanity check</h1><div class='pass'>{report['status']}</div><p>Pure read-only verification. New certificate calculations: 0.</p><h2>A. Cached-certificate provenance</h2>{provtab.to_html(index=False)}<pre>{json.dumps(comparisons,indent=2)}</pre><h2>B. Relative-width denominator audit</h2>{wtab.to_html(index=False)}<h2>C. Risk–coverage within K / residual / selection</h2>{rtab.to_html(index=False)}<h2>D. Top-10 false-mass coverage</h2><pre>{json.dumps(coverage,ensure_ascii=False,indent=2)}</pre><p>No simulation, ISTA, profile, certificate, feature, threshold, or v49 work was performed.</p>"
    (out/"v48_identity_sanity_check.html").write_text(html,encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o)))

if __name__=="__main__": main()
