# LegalHalluLens

**Typed hallucination auditing and calibrated multi-agent debate for trustworthy legal AI.**

[![arXiv](https://img.shields.io/badge/arXiv-2606.18021-b31b1b.svg)](https://arxiv.org/abs/2606.18021)
[![Hugging Face Papers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Papers-yellow)](https://huggingface.co/papers/2606.18021)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2606.18021-blue.svg)](https://doi.org/10.48550/arXiv.2606.18021)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](./LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/built%20with-LangGraph-1C3C3C.svg)](https://github.com/langchain-ai/langgraph)

📄 **Paper:** [arXiv:2606.18021](https://arxiv.org/abs/2606.18021) · [PDF](https://arxiv.org/pdf/2606.18021) · [Hugging Face Papers](https://huggingface.co/papers/2606.18021) · [DOI](https://doi.org/10.48550/arXiv.2606.18021)

Companion code for:

> Lalit Yadav, Akshaj Gurugubelli. *"LegalHalluLens: Typed Hallucination Auditing and Calibrated Multi-Agent Debate for Trustworthy Legal AI."* arXiv:2606.18021, 2026. To appear in the Proceedings of the 43rd International Conference on Machine Learning (ICML), Seoul, South Korea. PMLR 306, 2026.

**Authors:**
- Lalit Yadav — Independent Researcher, Sunnyvale, CA, USA — `lalitdv94@gmail.com` (correspondence)
- Akshaj Gurugubelli — Independent Researcher, San Diego, CA, USA

---

## Abstract

AI systems deployed in legal workflows hallucinate at rates that aggregate metrics report at ~52%, but this average conceals *where* errors concentrate and in *which direction* they run, leaving compliance officers without an actionable signal for trustworthy deployment. We present **LegalHalluLens**, an auditing framework with three components: **typed hallucination profiles** across four legally-motivated claim categories (numeric, temporal, obligation/entitlement, factual) over CUAD; a **Risk Direction Index (RDI)** that reduces omission-versus-invention bias to a single deployment-comparable scalar; and a **typed debate pipeline** calibrated to both magnitudes and directions. Across 510 contracts and 249,252 clause-level instances we measure a within-model gap of ≈38–40 pp between obligation/numeric and temporal claims that aggregate reporting hides, and show that two systems with matched 52% rates can carry opposite RDIs. The debate pipeline reduces fabricated detections by **45%** with per-category gains tracking the diagnosis, matching commercial APIs with a substantially smaller backbone (4B active parameters). The framework supports direction-aware procurement, accountability, and agent design for legal AI deployed in the wild.

## Highlights

- 🔬 **Typed, not aggregate.** A single ~52% hallucination rate hides a ≈38–40 pp within-model gap between claim types — surfaced here as per-type profiles.
- ↔️ **Direction matters.** Two systems with identical aggregate rates can fail in opposite directions (omission vs. invention); RDI separates them in one signed scalar.
- 🤝 **Calibrated debate beats generic debate.** A 6-role typed LangGraph pipeline cuts fabricated detections by **45%**, with a 4B-active-parameter open backbone matching commercial APIs.
- 📊 **Reproducible.** All extraction, judging, debate, and reporting steps ship as runnable notebooks over public CUAD v1.0.

---

## What this repo contains

This repository releases the **code, prompts, and analysis pipeline** behind LegalHalluLens. It does **not** ship the CUAD extraction outputs, judge labels, or aggregated result tables — those are derived artifacts that can be reproduced by running the pipeline against [CUAD v1.0](https://www.atticusprojectai.org/cuad) with the LLMs of your choice.

The framework has three components:

1. **Typed hallucination profiles** — stratify hallucination rates across four legally-motivated claim categories (numeric, temporal, obligation/entitlement, factual) instead of reporting a single aggregate rate.
2. **Risk Direction Index (RDI)** — a signed scalar that separates *omission* failures (dropping real obligations) from *invention* failures (asserting ones that don't exist), even when two systems score identically on aggregate hallucination rate.
3. **Typed multi-agent debate pipeline** — a 6-role LangGraph state machine (Skeptic, Supporter, Re-extractor, Arbiter, Verifier, Judge) operating on a baseline extraction, whose Skeptic challenges and asymmetric Add/Delete gates are calibrated from the measured per-type failure profile rather than chosen generically.

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

## Links & coverage

| | |
|---|---|
| 📄 Paper (arXiv) | https://arxiv.org/abs/2606.18021 |
| 📥 PDF | https://arxiv.org/pdf/2606.18021 |
| 🤗 Hugging Face Papers | https://huggingface.co/papers/2606.18021 |
| 🔗 DOI | https://doi.org/10.48550/arXiv.2606.18021 |
| 💻 Code | https://github.com/lalitdv9/LegalHallulens |
| 📰 The Model Wire | https://themodelwire.com/article/legalhallulens-typed-hallucination-auditing-and-calibrated-multi-agent-debate-fo-01KV9PW34XV72K5CPVGHG49JZF |
| 📰 gist.science | https://gist.science/paper/2606.18021 |

> **Note on Papers with Code:** Papers with Code was sunset by Meta in 2025; **[Hugging Face Papers](https://huggingface.co/papers/2606.18021)** is its successor and is where this paper's code/leaderboard links now live. If you upvote, follow, or claim authorship of the paper there, please link back to this repository.

---

## Contributing

Contributions, issues, and reproduction reports are welcome. Please see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for how to set up the environment, file issues, and open pull requests.

---

## License

Apache 2.0. See [`LICENSE`](./LICENSE).

The gemma-4-26B-A4B backbone used in Experiment 2 is released by Google under Apache 2.0; CUAD is released by The Atticus Project under CC BY 4.0.

---

## Citation

If you use LegalHalluLens, please cite the paper. A machine-readable [`CITATION.cff`](./CITATION.cff) is also provided (GitHub renders a "Cite this repository" button from it).

```bibtex
@article{yadav2026legalhallulens,
  title         = {LegalHalluLens: Typed Hallucination Auditing and Calibrated Multi-Agent Debate for Trustworthy Legal AI},
  author        = {Yadav, Lalit and Gurugubelli, Akshaj},
  journal       = {arXiv preprint arXiv:2606.18021},
  year          = {2026},
  eprint        = {2606.18021},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2606.18021},
  url           = {https://arxiv.org/abs/2606.18021},
}
```
