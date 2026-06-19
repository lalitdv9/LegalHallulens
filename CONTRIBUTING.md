# Contributing to LegalHalluLens

Thanks for your interest in LegalHalluLens — the companion code for
[*"LegalHalluLens: Typed Hallucination Auditing and Calibrated Multi-Agent Debate for Trustworthy Legal AI"*](https://arxiv.org/abs/2606.18021)
(arXiv:2606.18021).

Contributions, bug reports, and reproduction results are all welcome.

## Ways to contribute

- **Reproductions.** Ran the pipeline on CUAD with a different model or judge? Open an issue with your typed profiles, RDI values, and config so others can compare.
- **Bug reports.** Found a problem in the extraction, judge, debate, or reporting notebooks? File an issue with steps to reproduce.
- **New backbones / judges.** Add a client wrapper in [`utils/llm.py`](./utils/llm.py) and document it in a PR.
- **Documentation.** Clarifications, typo fixes, and better setup instructions are appreciated.

## Development setup

```bash
git clone https://github.com/lalitdv9/LegalHallulens.git
cd LegalHallulens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

API keys are read from environment variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`,
`OPENROUTER_API_KEY`, etc.). Place them in a local `.env` file (git-ignored) or
export them in your shell.

The notebooks expect CUAD v1.0 to be available locally and write per-run
artifacts to `Notebooks/batch_results/` (git-ignored). See the
[README](./README.md#running-the-pipeline) for the run order.

## Pull requests

1. Fork the repo and create a feature branch off `main`.
2. Keep changes focused; match the style and structure of the surrounding code.
3. If your change affects results or metrics, note what you ran it against.
4. Open a PR with a clear description of the motivation and the change.

## Reporting issues

Please include:

- What you ran (notebook, model, judge, dataset version).
- What you expected vs. what happened.
- Relevant logs or tracebacks (redact API keys).

## Code of conduct

Be respectful and constructive. This is a research artifact maintained on a
best-effort basis; please be patient with response times.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache 2.0 License](./LICENSE) that covers this project.
