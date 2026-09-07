"""
Pieces shared by every Eden AI endpoint: credentials, the exception class, and the per-request
`cost` Eden reports at the top level of each response body.
"""

from typing import Final

from pydantic import BaseModel, ValidationError

import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders

EDENAI_API_BASE: Final = "https://api.edenai.run/v3"


class EdenAIException(BaseLLMException):
    pass


class _EdenAIExtras(BaseModel):
    cost: float | None = None


def resolve_api_base(api_base: str | None) -> str:
    return api_base or get_secret_str("EDENAI_API_BASE") or EDENAI_API_BASE


def resolve_api_key(api_key: str | None) -> str | None:
    return api_key or get_secret_str("EDENAI_API_KEY")


def require_api_key(api_key: str | None, model: str) -> str:
    resolved: Final = resolve_api_key(api_key or litellm.api_key)
    if resolved is None:
        raise AuthenticationError(
            message="Missing Eden AI API key: set EDENAI_API_KEY or pass api_key",
            llm_provider=LlmProviders.EDENAI.value,
            model=model,
        )
    return resolved


def reported_cost(payload: object) -> float | None:
    try:
        extras: Final = (
            _EdenAIExtras.model_validate_json(payload)
            if isinstance(payload, bytes)
            else _EdenAIExtras.model_validate(payload)
        )
    except ValidationError:
        return None
    return extras.cost
