import pytest

import litellm


@pytest.fixture
def eden_key(monkeypatch) -> str:
    monkeypatch.delenv("EDENAI_API_BASE", raising=False)
    monkeypatch.setenv("EDENAI_API_KEY", "eden-test-key")
    monkeypatch.setattr(litellm, "api_key", None)
    return "eden-test-key"


@pytest.fixture
def httpx_transport(monkeypatch):
    """respx fakes httpx, so the async client must not sit on LiteLLM's default aiohttp transport."""
    monkeypatch.setattr(  # test-quality-ok: respx needs HTTPX enabled to fake the provider HTTP boundary.
        litellm,
        "disable_aiohttp_transport",
        True,
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()
