# Resume bullets

- Engineered an end-to-end Python product search platform over 42,994 WANDS products and 480 queries, combining TF-IDF, 384-dimensional FastEmbed vectors, hybrid fusion, logistic-regression reranking, FastAPI, Streamlit, and SQLite.
- Improved held-out judged-candidate nDCG@10 from 0.743282 for TF-IDF to 0.827633 for reranked hybrid retrieval across 72 test queries using leakage-safe query splits and graded relevance evaluation.
- Profiled 100 observations per component on the full 42,994-product index, measuring 21.604 ms lexical p50, 2.758 ms exact semantic-retrieval p50, and 53.183 ms loopback FastAPI p50 for the default reranked search.
