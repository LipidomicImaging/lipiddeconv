#!/usr/bin/env python3
"""One-pass, read-only analysis of the frozen 48-case v48 truth pilot."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


FEATURES = {
    "profile_relative_width": "higher_risk",
    "profile_lower": "higher_confidence",
    "rho_zero": "higher_confidence",
    "necessity_signal": "higher_confidence",
    "necessity_fit": "higher_confidence",
    "global_fragment_cone_residual": "higher_confidence",
    "reconstruction_relative_residual": "higher_risk",
    "group_relative_width": "higher_risk",
    "ista_nnls_disagreement": "higher_risk",
    "d_frag_observed": "higher_confidence",
    "diagnostic_energy_coverage": "higher_confidence",
}
PRIMARY = ["profile_relative_width", "rho_zero", "necessity_signal", "global_fragment_cone_residual"]


def auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    good = np.isfinite(score) & np.isfinite(y)
    y, score = y[good].astype(int), score[good]
    n1, n0 = int(y.sum()), int((1-y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(score)[y == 1].sum() - n1*(n1+1)/2) / (n1*n0))


def safe_spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.unique(x[m]).size < 2 or np.unique(y[m]).size < 2:
        return np.nan, np.nan
    r = spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue)


def quintiles(s: pd.Series) -> pd.Series:
    valid = s.notna() & np.isfinite(s)
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    if valid.sum() and s[valid].nunique() > 1:
        q = pd.qcut(s[valid], 5, labels=False, duplicates="drop")
        out.loc[valid] = q.astype("Int64") + 1
    return out


def fig_uri(fig) -> str:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def plot_error_k(active):
    d = active.groupby("K").relative_error.agg(["median", lambda x: x.quantile(.25), lambda x: x.quantile(.75)]).reset_index()
    fig, ax = plt.subplots(figsize=(6,4)); ax.plot(d.K, d["median"], marker="o")
    ax.fill_between(d.K, d["<lambda_0>"], d["<lambda_1>"], alpha=.25)
    ax.set(xlabel="K", ylabel="true-active relative error", title="True relative error versus K"); ax.grid(alpha=.25)
    return fig_uri(fig)


def plot_bad_k(active):
    d = active.assign(bad=active.relative_error>.2).groupby(["K","residual_ratio","selection_mode"]).bad.mean().reset_index()
    fig, ax = plt.subplots(figsize=(7,4))
    for (r,s), g in d.groupby(["residual_ratio","selection_mode"]):
        ax.plot(g.K, g.bad, marker="o", label=f"residual={r:g}, {s}")
    ax.set(xlabel="K", ylabel="fraction(error >20%)", ylim=(0,1), title="Error-rate versus K"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    return fig_uri(fig)


def plot_quintile(assoc, feature, title):
    d = assoc[(assoc.row_type=="quintile") & (assoc.feature==feature) & (assoc.stratum=="overall")]
    if d.empty: return None
    fig, ax = plt.subplots(figsize=(6,4)); ax.plot(d.quintile, d.bad20_fraction, marker="o")
    ax.set(xlabel="feature quintile (low to high)", ylabel="fraction(error >20%)", ylim=(0,1), title=title); ax.grid(alpha=.25)
    return fig_uri(fig)


def plot_group(groups):
    d = groups[groups.row_type=="overall"]
    if d.empty: return None
    fig, ax = plt.subplots(figsize=(5,4)); vals=[d.member_error_median.iloc[0], d.group_error_median.iloc[0]]
    ax.bar(["member", "group total"], vals); ax.set(ylabel="median relative error", title="Hard competition: member vs group")
    return fig_uri(fig)


def plot_auc(assoc):
    d = assoc[(assoc.row_type=="association") & (assoc.stratum=="overall")].dropna(subset=["direction_corrected_auroc"])
    if d.empty: return None
    d=d.sort_values("direction_corrected_auroc")
    fig, ax = plt.subplots(figsize=(7,max(3,.35*len(d)))); ax.barh(d.feature,d.direction_corrected_auroc); ax.axvline(.5,color="k",ls="--")
    ax.set(xlim=(.5,1), xlabel="direction-corrected AUROC", title="Single-feature prediction of error >20%")
    return fig_uri(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="v48_pilot_48"); ap.add_argument("--output",default=None)
    a=ap.parse_args(); root=Path(a.input); out=Path(a.output or root/"analysis"); out.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(root/"v48_pilot_manifest.csv")
    lip=pd.read_csv(root/"v48_pilot_lipid_observations.csv")
    core=["case_id","candidate_index","X_true","X_hat","profile_relative_width","rho_zero","global_fragment_cone_residual"]
    problems=[]
    expected=set(range(48)); got=set(manifest.case_id.astype(int)); cert=set(lip.case_id.astype(int))
    if got!=expected: problems.append(f"manifest missing cases {sorted(expected-got)}")
    if cert!=expected: problems.append(f"certificate missing cases {sorted(expected-cert)}")
    if len(lip)!=48*391: problems.append(f"lipid rows={len(lip)}, expected={48*391}")
    if lip.duplicated(["case_id","candidate_index"]).any(): problems.append("duplicate case_id+candidate_index")
    for c in core:
        if c not in lip: problems.append(f"missing core field {c}")
    if problems:
        raise SystemExit("CORE_INTEGRITY_FAIL: " + "; ".join(problems))
    meta=manifest[["case_id","K","residual_ratio","selection_mode","replicate","false_allocation_mass","reconstruction_relative_residual"]]
    drop=[c for c in meta.columns if c!="case_id" and c in lip.columns]
    d=lip.drop(columns=drop).merge(meta,on="case_id",how="left",validate="many_to_one")
    d["is_true"]=(d.X_true>0)
    active=d[d.is_true].copy(); active["absolute_error"]=(active.X_hat-active.X_true).abs(); active["relative_error"]=active.absolute_error/active.X_true; active["recovered_true_ratio"]=active.X_hat/active.X_true; active["bad20"]=active.relative_error>.2

    scen=[]
    for keys,g in active.groupby(["K","residual_ratio","selection_mode"],sort=True):
        cases=manifest[(manifest.K==keys[0])&(manifest.residual_ratio==keys[1])&(manifest.selection_mode==keys[2])]
        scen.append(dict(K=keys[0],residual_ratio=keys[1],selection_mode=keys[2],n_cases=cases.case_id.nunique(),n_true_active=len(g),relative_error_median=g.relative_error.median(),relative_error_p25=g.relative_error.quantile(.25),relative_error_p75=g.relative_error.quantile(.75),fraction_error_gt20=(g.relative_error>.2).mean(),fraction_error_gt50=(g.relative_error>.5).mean(),false_allocation_mass_median=cases.false_allocation_mass.median(),reconstruction_residual_median=cases.reconstruction_relative_residual.median()))
    scenario=pd.DataFrame(scen); scenario.to_csv(out/"v48_pilot_scenario_summary.csv",index=False)

    assoc=[]; available=[]; skipped=[]
    for f,direction in FEATURES.items():
        if f not in active.columns or active[f].notna().sum()<3: skipped.append(f); continue
        available.append(f)
        for stratum,g in [("overall",active)]+[(f"K={k}",q) for k,q in active.groupby("K")]:
            x=g[f].to_numpy(float); y=g.relative_error.to_numpy(float); bad=(y>.2).astype(int)
            rho,p=safe_spearman(x,y); raw=auc_rank(bad,x); corrected=max(raw,1-raw) if np.isfinite(raw) else np.nan
            assoc.append(dict(row_type="association",feature=f,direction=direction,stratum=stratum,n=np.isfinite(x).sum(),spearman=rho,spearman_p=p,raw_auroc=raw,direction_corrected_auroc=corrected))
            if stratum=="overall":
                q=g.copy(); q["quintile"]=quintiles(q[f])
                for qi,z in q.dropna(subset=["quintile"]).groupby("quintile"):
                    assoc.append(dict(row_type="quintile",feature=f,direction=direction,stratum=stratum,quintile=int(qi),n=len(z),median_relative_error=z.relative_error.median(),bad20_fraction=z.bad20.mean()))
                order=g.sort_values(f,ascending=(direction=="higher_risk"))
                for cov in [.2,.4,.6,.8,1.0]:
                    z=order.iloc[:max(1,int(np.ceil(cov*len(order))))]
                    assoc.append(dict(row_type="coverage",feature=f,direction=direction,stratum=stratum,coverage=cov,n=len(z),median_relative_error=z.relative_error.median(),bad20_fraction=z.bad20.mean()))
    association=pd.DataFrame(assoc); association.to_csv(out/"v48_pilot_feature_truth_association.csv",index=False)

    hard=d[(d.selection_mode=="hard_competition") & d.competition_group_id.notna()].copy()
    group_rows=[]
    if not hard.empty:
        totals=hard.groupby(["case_id","K","residual_ratio","competition_group_id"],as_index=False)[["X_true","X_hat"]].sum()
        totals=totals[totals.X_true>0].copy(); totals["group_error"]=(totals.X_hat-totals.X_true).abs()/totals.X_true
        members=hard[hard.X_true>0].copy(); members["member_error"]=(members.X_hat-members.X_true).abs()/members.X_true
        for label,mg,tg in [("overall",members,totals)]+[(f"K={k}",members[members.K==k],totals[totals.K==k]) for k in sorted(members.K.unique())]:
            group_rows.append(dict(row_type=label,n_members=len(mg),n_groups=len(tg),member_error_median=mg.member_error.median(),group_error_median=tg.group_error.median(),member_error_gt20_fraction=(mg.member_error>.2).mean(),group_error_gt20_fraction=(tg.group_error>.2).mean()))
    groups=pd.DataFrame(group_rows,columns=["row_type","n_members","n_groups","member_error_median","group_error_median","member_error_gt20_fraction","group_error_gt20_fraction"]); groups.to_csv(out/"v48_pilot_group_truth_summary.csv",index=False)

    overall=active.relative_error
    main_assoc=association[(association.row_type=="association")&(association.stratum=="overall")]
    signs={"profile_relative_width":1,"rho_zero":-1,"necessity_signal":-1,"global_fragment_cone_residual":-1}
    strong=[]; weak=[]
    for f,sgn in signs.items():
        r=main_assoc[main_assoc.feature==f]
        if r.empty: continue
        auc=float(r.direction_corrected_auroc.iloc[0]); rho=float(r.spearman.iloc[0])
        krows=association[(association.row_type=="association")&(association.feature==f)&association.stratum.str.startswith("K=")]
        stable=int(((krows.spearman*sgn)>0).sum())>=3
        if rho*sgn>0 and auc>=.60 and stable: strong.append(f)
        if rho*sgn>0 and auc>=.55: weak.append(f)
    conclusion="PROMISING" if strong else ("MIXED" if weak else "NOT_PROMISING")
    group_over=groups[groups.row_type=="overall"]
    group_better=(not group_over.empty and group_over.group_error_median.iloc[0] < group_over.member_error_median.iloc[0])
    factor_effects={
        "K_error_medians":active.groupby("K").relative_error.median().to_dict(),
        "residual_error_medians":active.groupby("residual_ratio").relative_error.median().to_dict(),
        "selection_error_medians":active.groupby("selection_mode").relative_error.median().to_dict(),
    }
    report={"status":conclusion,"integrity":{"cases":48,"lipid_rows":len(d),"true_active_rows":len(active),"problems":problems},"overall":{"median_relative_error":float(overall.median()),"p25":float(overall.quantile(.25)),"p75":float(overall.quantile(.75)),"fraction_error_gt20":float((overall>.2).mean()),"fraction_error_gt50":float((overall>.5).mean()),"median_case_false_allocation_mass":float(manifest.false_allocation_mass.median()),"median_reconstruction_residual":float(manifest.reconstruction_relative_residual.median())},"factor_effects":factor_effects,"available_features":available,"skipped_missing_features":skipped,"strong_stable_features":strong,"weak_features":weak,"group_total_better_than_member":bool(group_better)}
    (out/"v48_pilot_analysis_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    imgs=[plot_error_k(active),plot_bad_k(active)]
    for f,t in [("profile_relative_width","Profile width quintile versus error rate"),("rho_zero","rho_zero quintile versus error rate"),("necessity_signal","Necessity quintile versus error rate"),("global_fragment_cone_residual","Global cone quintile versus error rate")]:
        z=plot_quintile(association,f,t)
        if z: imgs.append(z)
    gz=plot_group(groups)
    if gz: imgs.append(gz)
    if len(imgs)<7:
        z=plot_auc(association)
        if z: imgs.append(z)
    best=main_assoc.sort_values("direction_corrected_auroc",ascending=False).head(1)
    best_text="none" if best.empty else f"{best.feature.iloc[0]} (AUROC={best.direction_corrected_auroc.iloc[0]:.3f}, Spearman={best.spearman.iloc[0]:.3f})"
    h=lambda x: pd.DataFrame(x).to_html(index=False,float_format=lambda v:f"{v:.4g}",border=0)
    html=f"""<!doctype html><meta charset='utf-8'><title>v48 truth pilot</title><style>body{{font-family:Arial;margin:32px;max-width:1200px}}table{{border-collapse:collapse}}th,td{{padding:5px 8px;border:1px solid #ddd}}img{{max-width:100%}}.status{{font-size:28px;font-weight:bold}}</style><h1>v48 48-case truth pilot</h1><div class='status'>{conclusion}</div><ol><li>总体 true-active median error={overall.median():.3f}；error&gt;20%={((overall>.2).mean()):.1%}；case false-allocation median={manifest.false_allocation_mass.median():.3f}。</li><li>K/residual/selection 的冻结分层结果见 scenario table；未训练模型、未寻找阈值。</li><li>最强现有单变量 certificate：{best_text}。</li><li>Profile width、necessity/rho_zero、global cone 的方向与 K 分层结果见 association table。</li><li>Observed cone 与 ISTA–NNLS disagreement 当前 pilot 输出不存在，未补算，不能声称有增量价值。</li><li>Hard competition 的 group total 是否优于 member：{group_better}。</li><li>是否值得扩大正式 v48：结论为 {conclusion}；依据为冻结单变量证据，不修改 feature。</li></ol><h2>Scenario summary</h2>{h(scenario)}<h2>Overall feature associations</h2>{h(main_assoc)}<h2>Group truth summary</h2>{h(groups)}<h2>Figures</h2>{''.join(f'<img src="{u}">' for u in imgs)}"""
    (out/"v48_pilot_analysis_report.html").write_text(html,encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
