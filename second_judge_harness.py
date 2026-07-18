"""
LegalHalluLens — Second-Judge Reliability Harness
==================================================
Re-judges existing extraction outputs with a second, architecturally-distinct
LLM judge, then measures inter-judge agreement on the two labels that matter:

  (1) binary verdict       -> claim_status SUPPORTED/CONTRADICTED (drives HalTP/HalGen)
  (2) directional mismatch -> judge_mismatch_type                  (drives RDI)

Runs on gpt-5.2 and qwen3-32b only — these are the two models whose RDI
separation (+0.161 vs -0.202) is the paper's headline directional claim.
Showing that separation survives a judge swap (gemini -> claude) is the goal.

Filters to 120-contract matched subset, run_id=1 only for consistency.

Run from repo root: python second_judge_harness.py
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

from utils.llm_func import judge_semantic_match
from utils.clauses_prompts import CLAUSE_TO_TYPE

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SECOND_JUDGE_MODEL = "anthropic/claude-sonnet-4-5"

# 120-contract matched subset derived from gemma baseline
SUBSET_PKL = "Notebooks/extractions/extractions_df_google-gemma-4-26b-a4b-it_20260413_104151.pkl"

# Only gpt-5.2 and qwen3-32b — the two models that matter for RDI separation
JOBS = [
    dict(
        name="gpt-5.2",
        ext_pkl="Notebooks/extractions/extractions_df_openai-gpt-5.2_20260209_155756.pkl",
        primary_jdg_pkl="Notebooks/judge_results/judge_results_openai-gpt-5.2_judged-by_google-gemini-2.5-flash_20260209_191508.pkl",
        det_col="is_impossible_ai",
        ans_col="answer_ai",
    ),
    dict(
        name="qwen3-32b",
        ext_pkl="Notebooks/extractions/extractions_df_qwen-qwen3-32b_20260215_211937.pkl",
        primary_jdg_pkl="Notebooks/judge_results/judge_results_qwen-qwen3-32b_judged-by_google-gemini-2.5-flash_20260217_064917.pkl",
        det_col="is_impossible_ai",
        ans_col="answer_ai",
    ),
]

OUT_DIR = Path("Notebooks/second_judge_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_EVERY = 200

# ──────────────────────────────────────────────────────────────────────────────
# Load 120-contract subset titles once
# ──────────────────────────────────────────────────────────────────────────────
print("Loading 120-contract subset titles...")
_subset_df = pd.read_pickle(SUBSET_PKL)
SUBSET_TITLES = set(_subset_df["title"].unique())
print(f"Subset: {len(SUBSET_TITLES)} contracts\n")

# ──────────────────────────────────────────────────────────────────────────────
# Re-judging
# ──────────────────────────────────────────────────────────────────────────────
def rejudge_extraction(ext_df, det_col, ans_col, model, ckpt_path):
    """Judge every TP clause in the 120-contract subset, run_id=1 only."""
    work = ext_df[
        (ext_df["is_impossible"] == False) &
        (ext_df[det_col] == False)
    ].copy()

    # Filter to 120-contract matched subset, run_id=1 only
    run_col = "run_id" if "run_id" in work.columns else None
    if run_col:
        work = work[work["title"].isin(SUBSET_TITLES) & (work[run_col] == 1)]
    else:
        work = work[work["title"].isin(SUBSET_TITLES)]

    print(f"  TP clauses in 120-contract subset (run_id=1): {len(work)}")

    work["judge_key"] = (
        work["title"].astype(str) + "||" +
        work["clause_name"].astype(str) + "||" +
        (work[run_col].astype(str) if run_col else "1")
    )

    # Resume from checkpoint if exists
    done = {}
    if Path(ckpt_path).exists():
        prev = pd.read_pickle(ckpt_path)
        done = {r["judge_key"]: r for r in prev.to_dict("records")}
        remaining = len(work) - len(done)
        print(f"  resume: {len(done)} already judged, {remaining} remaining")

    rows, since = [], 0
    total = len(work)
    for i, (_, r) in enumerate(work.iterrows()):
        if r["judge_key"] in done:
            rows.append(done[r["judge_key"]])
            continue
        ai = r[ans_col]
        gt = r["answers"]
        for attempt in range(3):
            try:
                is_match, reason, mm, jmodel = judge_semantic_match(
                    r["clause_name"], str(ai), str(gt), model=model)
                break
            except Exception as e:
                if attempt == 2:
                    is_match, reason, mm, jmodel = False, f"err:{e}", "evaluation_error", model
                else:
                    time.sleep(2 * (attempt + 1))
        rec = r.to_dict()
        rec.update(dict(
            llm_match=bool(is_match),
            judge_reason=reason,
            judge_mismatch_type=mm,
            claim_status="SUPPORTED" if is_match else "CONTRADICTED",
            judge_model=jmodel,
        ))
        rows.append(rec)
        since += 1
        if i % 50 == 0:
            status = "SUPPORTED" if is_match else "CONTRADICTED"
            print(f"  [{i+1}/{total}] {r['clause_name'][:40]} -> {status}")
        if since >= CHECKPOINT_EVERY:
            pd.DataFrame(rows).to_pickle(ckpt_path)
            since = 0
            print(f"  checkpoint saved @ {len(rows)} rows")

    out = pd.DataFrame(rows)
    out.to_pickle(ckpt_path)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Agreement metrics
# ──────────────────────────────────────────────────────────────────────────────
def cohens_kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return np.nan
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = sum((m[i, :].sum() / n) * (m[:, i].sum() / n) for i in range(len(labels)))
    return np.nan if pe == 1 else (po - pe) / (1 - pe)

def bootstrap_kappa_ci(a, b, n_boot=2000, seed=42):
    a, b = np.array(list(a)), np.array(list(b))
    n = len(a)
    if n < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ks = [cohens_kappa(a[rng.integers(0, n, n)], b[rng.integers(0, n, n)])
          for _ in range(n_boot)]
    ks = [k for k in ks if not np.isnan(k)]
    return (
        (round(float(np.percentile(ks, 2.5)), 3),
         round(float(np.percentile(ks, 97.5)), 3))
        if ks else (np.nan, np.nan)
    )

def rdi_from_judge(jdg, clause_type="ALL"):
    sub = jdg.copy()
    if "title" in sub.columns:
        sub = sub[sub["title"].isin(SUBSET_TITLES)]
    if clause_type != "ALL" and "clause_type" in sub.columns:
        sub = sub[sub["clause_type"] == clause_type]
    c = sub[sub["claim_status"] == "CONTRADICTED"]
    total = len(c)
    if total == 0:
        return np.nan
    extra = (c["judge_mismatch_type"] == "extra_condition").sum()
    missing = (c["judge_mismatch_type"] == "missing_condition").sum()
    return round((extra - missing) / total, 3)

def align(primary, second):
    if "title" in primary.columns:
        primary = primary[primary["title"].isin(SUBSET_TITLES)]
    keys = [k for k in ["title", "clause_name", "run_id"]
            if k in primary.columns and k in second.columns]
    p = primary[keys + ["claim_status", "judge_mismatch_type"]].rename(
        columns={"claim_status": "status_P", "judge_mismatch_type": "mm_P"})
    s = second[keys + ["claim_status", "judge_mismatch_type"]].rename(
        columns={"claim_status": "status_S", "judge_mismatch_type": "mm_S"})
    return p.merge(s, on=keys, how="inner")

# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────
def run(jobs=None):
    jobs = jobs or JOBS
    summary = []
    for job in jobs:
        print(f"\n{'='*60}")
        print(f"JOB: {job['name']}")
        print(f"{'='*60}")

        ext = pd.read_pickle(job["ext_pkl"])
        if "clause_type" not in ext.columns:
            ext["clause_type"] = ext["clause_name"].map(CLAUSE_TO_TYPE)

        primary = pd.read_pickle(job["primary_jdg_pkl"])
        if "clause_type" not in primary.columns:
            primary["clause_type"] = primary["clause_name"].map(CLAUSE_TO_TYPE)

        print(f"  extraction rows (full pkl): {len(ext)}")
        print(f"  primary judge rows (full pkl): {len(primary)}")

        ckpt = OUT_DIR / f"second_judge_{job['name']}.pkl"
        second = rejudge_extraction(
            ext, job["det_col"], job["ans_col"], SECOND_JUDGE_MODEL, ckpt)

        final_path = OUT_DIR / f"second_judge_{job['name']}_FINAL.pkl"
        second.to_pickle(final_path)
        print(f"  saved -> {final_path}")

        m = align(primary, second)
        both_c = m[
            (m["status_P"] == "CONTRADICTED") &
            (m["status_S"] == "CONTRADICTED")
        ]

        def dirlabel(x):
            return x if x in ("missing_condition", "extra_condition") else "other"

        k_bin  = cohens_kappa(m["status_P"], m["status_S"])
        ci_bin = bootstrap_kappa_ci(m["status_P"], m["status_S"])
        k_dir  = cohens_kappa(
            both_c["mm_P"].map(dirlabel),
            both_c["mm_S"].map(dirlabel))
        ci_dir = bootstrap_kappa_ci(
            both_c["mm_P"].map(dirlabel),
            both_c["mm_S"].map(dirlabel))

        rdi_P = rdi_from_judge(primary)
        rdi_S = rdi_from_judge(second)

        print(f"\n  RESULTS for {job['name']} (120-contract subset, run_id=1):")
        print(f"  aligned rows        : {len(m)}")
        print(f"  both-contradicted   : {len(both_c)}")
        print(f"  kappa binary        : {k_bin:.3f}  CI{ci_bin}")
        print(f"  kappa directional   : {k_dir:.3f}  CI{ci_dir}")
        print(f"  RDI gemini (primary): {rdi_P:+.3f}")
        print(f"  RDI claude (second) : {rdi_S:+.3f}")
        print(f"  RDI delta           : {rdi_S - rdi_P:+.3f}")

        summary.append(dict(
            job=job["name"],
            n_aligned=len(m),
            n_both_contradicted=len(both_c),
            kappa_binary=round(k_bin, 3),
            kappa_binary_ci=str(ci_bin),
            kappa_directional=round(k_dir, 3),
            kappa_directional_ci=str(ci_dir),
            rdi_primary=rdi_P,
            rdi_second=rdi_S,
            rdi_delta=round(rdi_S - rdi_P, 3),
        ))

    summary_df = pd.DataFrame(summary)
    summary_path = OUT_DIR / "second_judge_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"COMPLETE. Summary -> {summary_path}")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    return summary_df


if __name__ == "__main__":
    # Runs gpt-5.2 and qwen3-32b on the 120-contract matched subset
    # These are the two models whose RDI separation the paper claims
    run()