"""
Fix bootstrap CI computation and recompute from saved second-judge results.
Run from repo root: python fix_kappa_ci.py
No API calls — pure math on saved pkl files.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("Notebooks/second_judge_results")
SUBSET_PKL = "Notebooks/extractions/extractions_df_google-gemma-4-26b-a4b-it_20260413_104151.pkl"

# Load subset titles
_subset_df = pd.read_pickle(SUBSET_PKL)
SUBSET_TITLES = set(_subset_df["title"].unique())

JOBS = [
    dict(name="gpt-5.2",
         primary_jdg_pkl="Notebooks/judge_results/judge_results_openai-gpt-5.2_judged-by_google-gemini-2.5-flash_20260209_191508.pkl"),
    dict(name="qwen3-32b",
         primary_jdg_pkl="Notebooks/judge_results/judge_results_qwen-qwen3-32b_judged-by_google-gemini-2.5-flash_20260217_064917.pkl"),
    dict(name="gemini-3-flash",
         primary_jdg_pkl="Notebooks/judge_results/judge_results_google-gemini-3-flash-preview_judged-by_google-gemini-2.5-flash_20260201_164538.pkl"),
    dict(name="llama-3.3-70b",
         primary_jdg_pkl="Notebooks/judge_results/judge_results_meta-llama-llama-3.3-70b-instruct_judged-by_google-gemini-2.5-flash_20260201_054729.pkl"),
]

def cohens_kappa(a, b):
    a, b = np.array(list(a)), np.array(list(b))
    n = len(a)
    if n == 0:
        return np.nan
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = sum((m[i,:].sum()/n)*(m[:,i].sum()/n) for i in range(len(labels)))
    return np.nan if pe == 1 else (po - pe) / (1 - pe)

def bootstrap_kappa_ci(a, b, n_boot=2000, seed=42):
    """Fixed bootstrap — samples PAIRED indices, not independent."""
    a, b = np.array(list(a)), np.array(list(b))
    n = len(a)
    if n < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        # Sample the SAME indices for both — paired bootstrap
        idx = rng.integers(0, n, n)
        ks.append(cohens_kappa(a[idx], b[idx]))
    ks = [k for k in ks if not np.isnan(k)]
    if not ks:
        return (np.nan, np.nan)
    return (round(float(np.percentile(ks, 2.5)), 3),
            round(float(np.percentile(ks, 97.5)), 3))

def rdi_from_df(df, subset_titles):
    sub = df[df["title"].isin(subset_titles)] if "title" in df.columns else df
    c = sub[sub["claim_status"] == "CONTRADICTED"]
    total = len(c)
    if total == 0:
        return np.nan
    extra = (c["judge_mismatch_type"] == "extra_condition").sum()
    missing = (c["judge_mismatch_type"] == "missing_condition").sum()
    return round((extra - missing) / total, 3)

def dirlabel(x):
    return x if x in ("missing_condition", "extra_condition") else "other"

summary = []
for job in JOBS:
    name = job["name"]
    final_path = OUT_DIR / f"second_judge_{name}_FINAL.pkl"
    if not final_path.exists():
        print(f"MISSING: {final_path}")
        continue

    second = pd.read_pickle(final_path)
    primary = pd.read_pickle(job["primary_jdg_pkl"])
    if "clause_type" not in primary.columns:
        from utils.clauses_prompts import CLAUSE_TO_TYPE
        primary["clause_type"] = primary["clause_name"].map(CLAUSE_TO_TYPE)

    # Filter primary to subset
    if "title" in primary.columns:
        primary_sub = primary[primary["title"].isin(SUBSET_TITLES)]
    else:
        primary_sub = primary

    # Align
    keys = [k for k in ["title", "clause_name", "run_id"]
            if k in primary_sub.columns and k in second.columns]
    p = primary_sub[keys + ["claim_status", "judge_mismatch_type"]].rename(
        columns={"claim_status": "status_P", "judge_mismatch_type": "mm_P"})
    s = second[keys + ["claim_status", "judge_mismatch_type"]].rename(
        columns={"claim_status": "status_S", "judge_mismatch_type": "mm_S"})
    m = p.merge(s, on=keys, how="inner")
    both_c = m[(m["status_P"] == "CONTRADICTED") & (m["status_S"] == "CONTRADICTED")]

    # Fixed kappa with correct paired bootstrap
    k_bin  = cohens_kappa(m["status_P"], m["status_S"])
    ci_bin = bootstrap_kappa_ci(m["status_P"], m["status_S"])
    k_dir  = cohens_kappa(both_c["mm_P"].map(dirlabel), both_c["mm_S"].map(dirlabel))
    ci_dir = bootstrap_kappa_ci(both_c["mm_P"].map(dirlabel), both_c["mm_S"].map(dirlabel))

    rdi_P = rdi_from_df(primary_sub, SUBSET_TITLES)
    rdi_S = rdi_from_df(second, SUBSET_TITLES)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  aligned rows        : {len(m)}")
    print(f"  both-contradicted   : {len(both_c)}")
    print(f"  kappa binary        : {k_bin:.3f}  95% CI {ci_bin}")
    print(f"  kappa directional   : {k_dir:.3f}  95% CI {ci_dir}")
    print(f"  RDI gemini (primary): {rdi_P:+.3f}")
    print(f"  RDI claude (second) : {rdi_S:+.3f}")
    print(f"  RDI delta           : {rdi_S - rdi_P:+.3f}")

    summary.append(dict(
        model=name,
        n_aligned=len(m),
        n_both_contradicted=len(both_c),
        kappa_binary=round(k_bin, 3),
        kappa_binary_ci_lo=ci_bin[0],
        kappa_binary_ci_hi=ci_bin[1],
        kappa_directional=round(k_dir, 3),
        kappa_dir_ci_lo=ci_dir[0],
        kappa_dir_ci_hi=ci_dir[1],
        rdi_gemini=rdi_P,
        rdi_claude=rdi_S,
        rdi_delta=round(rdi_S - rdi_P, 3),
    ))

df_out = pd.DataFrame(summary)
out_path = OUT_DIR / "second_judge_summary_FIXED.csv"
df_out.to_csv(out_path, index=False)
print(f"\n{'='*60}")
print("FIXED SUMMARY")
print(f"{'='*60}")
print(df_out.to_string(index=False))
print(f"\nSaved -> {out_path}")
