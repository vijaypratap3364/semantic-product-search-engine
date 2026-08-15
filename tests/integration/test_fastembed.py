"""Optional local-only FastEmbed integration coverage."""

from __future__ import annotations

import numpy as np
import pytest

from product_search.config import load_settings
from product_search.indexing.dense import FastEmbedProvider, normalize_embeddings


@pytest.mark.embedding
def test_configured_fastembed_model_when_available_locally() -> None:
    settings = load_settings()
    try:
        provider = FastEmbedProvider(
            settings.dense.model_name,
            cache_dir=settings.paths.embeddings / "model_cache",
            local_files_only=True,
        )
        vectors = list(provider.embed_queries(["round coffee table"], batch_size=1))
    except Exception as error:
        pytest.skip(f"configured FastEmbed model is not present in the local cache: {error}")

    assert provider.model_name == "BAAI/bge-small-en-v1.5"
    assert len(vectors) == 1
    normalized = normalize_embeddings(
        np.asarray(vectors, dtype=np.float32),
        expected_dimension=settings.dense.expected_dimension,
    )
    assert normalized.shape == (1, 384)
    assert np.linalg.norm(normalized[0]) == pytest.approx(1.0, abs=1e-5)
