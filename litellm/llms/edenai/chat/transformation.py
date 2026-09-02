"""
Support for OpenAI's `/v1/chat/completions` endpoint on Eden AI.

Eden AI is an OpenAI-compatible gateway (one key across 1000+ models), so requests go through the
shared HTTP handler untouched. Every Eden response reports the real per-request cost at the top
level of the body; the only translation here lifts that number into LiteLLM's cost tracking.

Docs: https://www.edenai.co/docs
"""

from collections.abc import AsyncIterator, Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIChatCompletionStreamingHandler, OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage

from ..common_utils import EdenAIException

if TYPE_CHECKING:
    import tiktoken

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

EDENAI_API_BASE: Final = "https://api.edenai.run/v3"
_RESPONSE_COST_HEADER: Final = "llm_provider-x-litellm-response-cost"
_OPTIONAL_MAPPING: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)


class _EdenAIExtras(BaseModel):
    cost: float | None = None


class _EdenAIModel(BaseModel):
    id: str


class _EdenAIModelCatalog(BaseModel):
    data: tuple[_EdenAIModel, ...]


def _reported_cost(payload: object) -> float | None:
    try:
        extras: Final = (
            _EdenAIExtras.model_validate_json(payload)
            if isinstance(payload, bytes)
            else _EdenAIExtras.model_validate(payload)
        )
    except ValidationError:
        return None
    return extras.cost


def _stream_options_with_usage(request: Mapping[str, object]) -> Mapping[str, object]:
    current: Final = _OPTIONAL_MAPPING.validate_python(request.get("stream_options")) or MappingProxyType({})
    return MappingProxyType({**current, "include_usage": True})


class EdenAIChatCompletionStreamingHandler(OpenAIChatCompletionStreamingHandler):
    def chunk_parser(self, chunk: dict[str, object]) -> ModelResponseStream:  # mutable-ok: inherited contract
        parsed: Final = super().chunk_parser(chunk)
        cost: Final = _reported_cost(chunk)
        usage: Final[object] = getattr(parsed, "usage", None)
        if cost is not None and isinstance(usage, Usage):
            usage.cost = cost
        return parsed


class EdenAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "edenai"

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: inherited contract
        return [*super().get_supported_openai_params(model), "reasoning_effort"]  # mutable-ok: inherited contract

    def _get_openai_compatible_provider_info(self, api_base: str | None, api_key: str | None) -> tuple[str, str | None]:
        return (
            api_base or get_secret_str("EDENAI_API_BASE") or EDENAI_API_BASE,
            api_key or get_secret_str("EDENAI_API_KEY"),
        )

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        headers: dict[str, object],  # mutable-ok: inherited contract
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        request: Final[dict[str, object]] = super().transform_request(  # mutable-ok: inherited contract
            model, messages, optional_params, litellm_params, headers
        )
        if not request.get("stream"):
            return request
        return {**request, "stream_options": dict(_stream_options_with_usage(request))}  # mutable-ok: JSON body

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict[str, object],  # mutable-ok: inherited contract
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        encoding: "tiktoken.Encoding | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        response: Final = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        cost: Final = _reported_cost(raw_response.content)
        if cost is None:
            return response
        hidden_params: Final[dict[str, object]] = response._hidden_params  # mutable-ok: plain dict by contract
        headers: Final = _OPTIONAL_MAPPING.validate_python(hidden_params.get("additional_headers"))
        hidden_params["additional_headers"] = {  # mutable-ok: hidden params are a plain dict by contract
            **(headers or MappingProxyType({})),
            _RESPONSE_COST_HEADER: cost,
        }
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return EdenAIException(message=error_message, status_code=status_code, headers=headers)

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> EdenAIChatCompletionStreamingHandler:
        return EdenAIChatCompletionStreamingHandler(
            streaming_response=streaming_response, sync_stream=sync_stream, json_mode=json_mode
        )

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: inherited contract
        resolved_base, _ = self._get_openai_compatible_provider_info(api_base, api_key)
        response: Final = litellm.module_level_client.get(url=f"{resolved_base}/models")
        if not response.is_success:
            raise EdenAIException(status_code=response.status_code, message=response.text, headers=response.headers)
        catalog: Final = _EdenAIModelCatalog.model_validate(response.json())
        return [f"edenai/{model.id}" for model in catalog.data]  # mutable-ok: inherited contract
