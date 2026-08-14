import numpy as np

from hydra_mem.embeddings import TfidfEmbedder, build_embedder


def test_tfidf_shape_and_normalization():
    emb = TfidfEmbedder(dim=128)
    vecs = emb.embed(["hello world", "another document"])
    assert vecs.shape == (2, 128)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0)


def test_tfidf_deterministic():
    emb = TfidfEmbedder(dim=128)
    assert np.allclose(emb.embed(["same text"]), emb.embed(["same text"]))


def test_tfidf_similarity_ordering():
    emb = TfidfEmbedder(dim=512)
    a, b, c = emb.embed(["the cat sat on the mat", "the cat sat on the mat today", "quantum physics lecture"])
    assert float(a @ b) > float(a @ c)


def test_build_embedder_tfidf():
    assert isinstance(build_embedder("tfidf"), TfidfEmbedder)


def test_build_embedder_unknown():
    import pytest

    with pytest.raises(ValueError, match="unknown embedding backend"):
        build_embedder("nope")
