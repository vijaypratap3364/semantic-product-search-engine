# 90-second interview demo

Use the real local application and generated reports. The suggested queries demonstrate specific capabilities, but describe only the results actually visible during the demo.

| Time | Action | Talk track |
|---:|---|---|
| 0–10s | Show the repository root and `src/product_search` packages. | “This is an end-to-end, CPU-friendly product search system: reproducible data ingestion, four retrieval approaches, leakage-controlled evaluation, API, dashboard, and feedback.” |
| 10–20s | Open the WANDS section of the README. | “It uses Wayfair's WANDS dataset: 42,994 products, 480 queries, and human Exact, Partial, and Irrelevant judgments. I preserve source labels and resolve duplicates in a separate canonical evaluation table.” |
| 20–32s | Search `round coffee table` in lexical mode. | “TF-IDF is the transparent baseline. It is especially strong when the query and catalog share discriminative product terms.” |
| 32–45s | Search `turquoise pillows` in semantic mode. | “FastEmbed maps query and product text into the same 384-dimensional space, so it can recover meaning even when wording differs. I still judge the result by what is actually shown here.” |
| 45–58s | Enable comparison mode for the same query. | “The dashboard calls FastAPI for lexical, semantic, and hybrid results side by side. It never loads a retrieval model directly.” |
| 58–68s | Open hybrid results and expand “Why this result?”. | “Hybrid uses validation-selected normalized score fusion: 10% lexical and 90% semantic. Explanations expose matched terms and component contributions without an LLM.” |
| 68–77s | Open the Benchmarks page. | “These values are loaded from the generated held-out report, never hardcoded. The selected reranker reached 0.827633 nDCG@10 across 72 untouched test queries.” |
| 77–84s | Open `http://127.0.0.1:8000/docs`. | “FastAPI exposes health, readiness, model metadata, search modes, structured search, feedback, and local aggregate analytics.” |
| 84–90s | Show the GitHub Actions badge or latest CI run. | “CI repeats formatting, linting, Linux-safe type checking, tests, coverage, and package-build checks using small fixtures—without downloading WANDS or model weights.” |

If a service or artifact is unavailable, do not substitute a screenshot or claim a result. Show the actionable readiness error, rebuild the missing artifact, and repeat the demo.
