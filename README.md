# LegalHalluLens

**Typed hallucination auditing and calibrated multi-agent debate for trustworthy legal AI.**

Companion code for the ICML 2026 AIWILD workshop paper:
*"LegalHalluLens: Typed Hallucination Auditing and Calibrated Multi-Agent Debate for Trustworthy Legal AI"* — Lalit Yadav, Akshaj Gurugubelli.

---

## What this repo contains

This repository releases the **code, prompts, and analysis pipeline** behind LegalHalluLens. It does **not** ship the CUAD extraction outputs, judge labels, or aggregated result tables — those are derived artifacts that can be reproduced by running the pipeline against [CUAD v1.0](https://www.atticusprojectai.org/cuad) with the LLMs of your choice.

The framework has three components:

1. **Typed hallucination profiles** — stratify hallucination rates across four legally-motivated claim categories (numeric, temporal, obligation/entitlement, factual) instead of reporting a single aggregate rate.
2. **Risk Direction Index (RDI)** — a signed scalar that separates *omission* failures (dropping real obligations) from *invention* failures (asserting ones that don't exist), even when two systems score identically on aggregate hallucination rate.
3. **Typed multi-agent debate pipeline** — a 7-role LangGraph state machine (Extractor → Skeptic/Supporter → Route → Re-extractor / Arbiter → Verifier → Judge) whose Skeptic challenges and asymmetric Add/Delete gates are calibrated from the measured per-type failure profile rather than chosen generically.

For headline results, the typed failure ordering, RDI values per model, and the gemma-debate composite leaderboard, see the paper.

---

## Repo layout

```
LegalHalluLens/
├── Notebooks/
│   ├── LegalHalluLens_BatchProcessing.ipynb     # Experiment 1: clause extraction over CUAD
│   ├── LegalHalluLens_JudgeProcessing.ipynb     # LLM-as-judge labelling (Appendix A rubric)
│   ├── LegalHalluLens_Debate_LangGraph.ipynb    # Experiment 2: typed debate pipeline
│   └── ModelComparison_Report.ipynb             # All tables and figures from the paper
├── utils/
│   ├── clauses_prompts.py        # Extraction prompts and clause-type taxonomy
│   ├── llm.py                    # LLM client wrappers (OpenAI, Gemini, OpenRouter, Ollama)
│   ├── llm_func.py               # Higher-level extraction / judge call helpers
│   └── helpers.py                # JSON parsing and shared utilities
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

`Notebooks/batch_results/` is excluded from version control. The notebooks read pickled extraction and judge outputs from that directory when present; users are expected to regenerate them locally.

---

## Models and judge

| Role | Model | Notes |
|---|---|---|
| Benchmark (Experiment 1) | `google/gemini-3-flash-preview` | Commercial API |
| Benchmark (Experiment 1) | `openai/gpt-5.2` | Commercial API |
| Benchmark (Experiment 1) | `qwen/qwen3-32b` | Open, 32.8B params |
| Benchmark (Experiment 1) | `meta-llama/llama-3.3-70b-instruct` | Open, 70B params |
| Mitigation backbone (Experiment 2) | `google/gemma-4-26b-a4b-it` | Open, MoE, 4B active params, Apache 2.0 |
| Evaluation judge | `google/gemini-2.5-flash` | Five-criterion rubric (Appendix A), temperature 0 |

The judge labels each detected clause as `supported` or `contradicted` and assigns a `mismatch_type` (`none`, `numeric`, `temporal`, `obligation`, `scope`, `missing_condition`, `extra_condition`, `other`). The `missing_condition` and `extra_condition` labels are what RDI is derived from.

---

## Key metrics

| Metric | Formula | What it captures |
|---|---|---|
| FAR | FP / (FP + TN) | Invents absent clauses |
| FRR | FN / (FN + TP) | Misses present clauses |
| HalTP | contradicted / TP | Content-error rate among detected clauses (Experiment 1) |
| HalGen | (contradicted + FP) / (TP + FP) | Content errors + fabrications among generated outputs (Experiment 2) |
| JEq | supported / (TP + FN) | End-to-end correctness against the CUAD oracle |
| RDI | (p_extra − p_missing) / 100 | Signed direction: negative → omits, positive → invents |

---

## Running the pipeline

Set up an environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

API keys are read from environment variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, etc.). The notebooks are organised so each one can be run end-to-end:

1. **`LegalHalluLens_BatchProcessing.ipynb`** — loads CUAD, runs clause extraction for the chosen model, and writes a per-clause extraction DataFrame to `Notebooks/batch_results/extractions/`.
2. **`LegalHalluLens_JudgeProcessing.ipynb`** — reads an extraction pickle, calls the judge against the CUAD oracle answers, and writes a judged DataFrame to `Notebooks/batch_results/judge_results/`.
3. **`LegalHalluLens_Debate_LangGraph.ipynb`** — given a baseline extraction artifact, runs the typed debate pipeline (Skeptic / Supporter / Route / Re-extractor / Arbiter / Verifier / Judge) and emits a debate extraction artifact for the same contracts.
4. **`ModelComparison_Report.ipynb`** — given the extraction and judge pickles for every model under study, computes every table and figure in the paper, including HalTP, HalGen, RDI with bootstrap CIs, the typed leaderboard, and the per-type debate deltas.

The notebooks expect the CUAD v1.0 corpus to be locally available; they were developed against the standard HuggingFace mirror.

---

## Caveats

- **Single-judge dependence.** Every reported metric in the paper (HalTP, HalGen, RDI) flows through `gemini-2.5-flash` applying the rubric in Appendix A. RDI is intended as a *directional* signal, not a cardinal measure of risk magnitude.
- **Scope errors** (62–71% of contradictions) carry no directional label, so RDI is computed on the 29–38% of content errors with clear directional character.
- **Experiment 2** is a single run on a 120-contract matched subset with the gemma-4-26B-A4B backbone; the composite ranking is evidence for that comparison only.
- **Scope of evidence.** Numerical results apply to 510 English-US commercial contracts in CUAD v1.0; the typed failure ordering is consistent across four architectures but transfer to other jurisdictions and document types is an open question.
- **Diagnostic, not clearance.** Even the best configuration in the paper contradicted the source on 58.6% of detected clause contents. Typed evaluation should inform — not replace — qualified human review in high-stakes legal workflows.

---

## License

Apache 2.0. See [`LICENSE`](./LICENSE).

The gemma-4-26B-A4B backbone used in Experiment 2 is released by Google under Apache 2.0; CUAD is released by The Atticus Project under CC BY 4.0.

---

## Citation

```bibtex
@inproceedings{yadav2026legalhallulens,
  title     = {LegalHalluLens: Typed Hallucination Auditing and Calibrated Multi-Agent Debate for Trustworthy Legal AI},
  author    = {Yadav, Lalit and Gurugubelli, Akshaj},
  booktitle = {ICML 2026 Workshop on Agents in the Wild (AIWILD)},
  year      = {2026},
}
```
