# Screenshot capture guide

No screenshots are committed until they can be captured from the real running application. This avoids presenting mock data or fabricated UI state as a working demo.

## Before capturing

1. Build the data, indexes, final evaluation report, and selected reranker by following the root README.
2. Start FastAPI with `uv run uvicorn product_search.api.main:app --reload`.
3. Start Streamlit in a second terminal with `uv run streamlit run src/product_search/ui/app.py`.
4. Confirm the Streamlit System page reports a healthy, ready API and the expected 42,994 indexed products.
5. Use a consistent desktop browser size, hide personal bookmarks or notifications, and do not alter result values in an image editor.

## Main search screen

1. Open the Search page.
2. Select `default`, choose top 5, and search for `round coffee table`.
3. Expand one “Why this result?” panel so the matched terms and component scores are visible.
4. Capture the project title, controls, and several real results.
5. Save the image as `docs/screenshots/main-search.png`.

## Comparison screen

1. On the Search page, enter `turquoise pillows`.
2. Enable comparison mode and run the query.
3. Show the lexical, semantic, and hybrid tabs or columns with their real responses.
4. Capture enough of the view to make the comparison controls and result differences clear.
5. Save the image as `docs/screenshots/comparison.png`.

## Benchmark screen

1. Open the Benchmarks page after generating `artifacts/reports/final_test_metrics.json`.
2. Confirm that lexical, semantic, hybrid, and reranked-hybrid metrics render from that report.
3. Capture the visible nDCG@10, Recall@10, MRR@10, and median latency values.
4. Save the image as `docs/screenshots/benchmark.png`.

After capture, verify each image against the live application, then add the images and corresponding README image links in a separate documentation commit.
