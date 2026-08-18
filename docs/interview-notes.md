# Interview notes

## What is semantic search?

Semantic search represents the query and documents as learned numeric embeddings, then ranks by vector similarity. Unlike exact token matching, nearby vectors can express related meaning even when the query and product use different words.

## Why TF-IDF first?

TF-IDF is fast to build, interpretable, CPU-friendly, and a strong baseline for exact product terminology. Starting there establishes whether the added complexity of embeddings and fusion produces a measured improvement.

## Why cosine similarity?

Search should compare direction—the pattern of features or meaning—rather than raw vector magnitude. TF-IDF uses L2 normalization, and the project explicitly L2-normalizes dense vectors, so their dot products equal cosine similarity.

## Why FastEmbed?

FastEmbed provides small ONNX-based embedding models without requiring PyTorch or a serving cluster. `BAAI/bge-small-en-v1.5` produced 384-dimensional English embeddings and kept the entire solution practical on ordinary CPU hardware.

## Why hybrid retrieval?

Lexical and semantic retrieval fail differently. Exact tokens matter for catalog constraints, while embeddings help with vocabulary mismatch. Hybrid retrieval normalizes per-query scores before combining them, and the 0.1 lexical/0.9 semantic weights were selected on validation nDCG@10—not guessed or tuned on test data.

## How does nDCG work conceptually?

nDCG rewards putting highly relevant products near the top. DCG assigns larger gain to Exact than Partial judgments and discounts gains at lower ranks; dividing by the best possible DCG for that query makes the score comparable on a zero-to-one scale.

## Why split by query?

Rows for one query are highly related. A row-level split could put judgments for the same query into training and evaluation, making performance look better than it generalizes. Query-level splits test behavior on entirely unseen search intents.

## How did you prevent evaluation leakage?

The seed-42 split assigns each query ID to exactly one of 336 train, 72 validation, or 72 test queries. The reranker learns only from train judgments; validation selects hybrid weights, classifier settings, and reranker eligibility. Test queries were used once after configurations were frozen, and the final report records source and artifact hashes.

## Why not use Elasticsearch?

The goal was to expose and compare the retrieval mechanics directly. Scikit-learn sparse matrices make TF-IDF, score normalization, and evaluation easy to inspect with no service infrastructure. Elasticsearch would be appropriate when production requirements include distributed indexing, filters, operational scaling, and mature text-analysis features.

## Why not use a vector database?

The catalog has 42,994 products. Exact NumPy scoring remains simple and measured at a 2.758 ms p50 for similarity plus top-K after query embedding. A vector database would add infrastructure without solving a measured bottleneck at this scale.

## When would NumPy no longer be enough?

I would reassess when exact scoring or memory misses a defined service objective under representative concurrency and catalog growth. At 384 float32 dimensions, raw embeddings alone require about 1.43 GiB per million products before metadata and working memory. I would benchmark ANN recall, latency, cost, and operational tradeoffs rather than choose a catalog-size threshold by intuition.

## How does the reranker work?

Hybrid retrieval first creates a top-100 pool. A `StandardScaler` plus three-class multinomial `LogisticRegression` predicts Irrelevant, Partial, and Exact using 13 features available for arbitrary future queries, including retrieval scores and ranks, token overlap, exact phrase evidence, and text lengths. The final score is `P(Partial) * 1 + P(Exact) * 2`, and only the existing pool is reordered.

## What did the reranker improve?

On validation queries, judged-candidate nDCG@10 rose from 0.791515 for hybrid to 0.805658 for reranked hybrid, so it passed the predeclared eligibility rule. On the frozen test set it raised nDCG@10 from 0.817523 to 0.827633, the best tested quality score, while Precision@10, Recall@10, and MRR@10 regressed slightly. The production choice is documented with both sides of that tradeoff.

## What were the biggest failure modes?

The held-out error analysis found vocabulary-dependent lexical misses, semantic losses on exact constraints, five failed tail queries, ten partial-versus-exact confusion cases, and mixed reranker behavior: it helped 18 queries and hurt 15. Unjudged catalog products also limit what offline full-catalog metrics can conclude.

## How would this scale to millions of products?

I would retain the offline/online boundaries but move catalog preparation and incremental index builds to scheduled jobs, partition metadata in durable storage, and benchmark an ANN index for dense candidate generation. Lexical retrieval could move to a distributed search engine, artifacts would use versioned promotion and rollback, and the API tier would scale horizontally with caches, rate limits, monitoring, and relevance-drift checks.

## How would personalization be added?

I would keep general relevance as the first-stage candidate generator, then add consented session or user features in a separate reranking layer: recent categories, brands, price affinity, and interaction history. Training and evaluation would use time-based splits, privacy controls, opt-out behavior, cold-start fallbacks, and checks that personalization does not erase query relevance or systematically disadvantage product groups.

## What did you personally implement?

I implemented the staged repository foundation; reproducible WANDS download, validation, hashing, and preprocessing; canonical judgment policy; query splits and hand-checked metrics; lexical, semantic, hybrid, and supervised ranking; artifact integrity checks; held-out evaluation and profiling; the unified service, FastAPI API, SQLite feedback, Streamlit client; fixture-based CI; and the documentation needed to reproduce and explain the decisions. I also recorded regressions and missing measurements rather than inventing favorable numbers.
