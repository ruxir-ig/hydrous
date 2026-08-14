import pytest

from hydrous.config import Settings, build_llm
from hydrous.llm import MockLLM, OpenAICompatibleLLM


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_openai_credentials():
    s = _settings(llm_provider="openai", openai_api_key="sk-test")
    base_url, key = s.resolve_credentials()
    assert base_url is None
    assert key == "sk-test"


def test_openrouter_preset():
    s = _settings(llm_provider="openrouter", openrouter_api_key="or-test")
    base_url, key = s.resolve_credentials()
    assert base_url == "https://openrouter.ai/api/v1"
    assert key == "or-test"


def test_custom_requires_base_url():
    s = _settings(llm_provider="custom", llm_api_key="k")
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        s.resolve_credentials()


def test_custom_endpoint():
    s = _settings(llm_provider="custom", llm_api_key="k", llm_base_url="http://localhost:8000/v1")
    base_url, key = s.resolve_credentials()
    assert (base_url, key) == ("http://localhost:8000/v1", "k")


def test_missing_key_names_env_var():
    s = _settings(llm_provider="openai")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        s.resolve_credentials()


def test_unknown_provider():
    s = _settings(llm_provider="wat")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        s.resolve_credentials()


def test_build_llm_mock():
    assert isinstance(build_llm(_settings(llm_provider="mock")), MockLLM)
    assert isinstance(build_llm(_settings(llm_provider="openai"), force_mock=True), MockLLM)


def test_build_llm_real():
    llm = build_llm(_settings(llm_provider="openai", openai_api_key="sk-test", llm_model="gpt-x"))
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.model == "gpt-x"
