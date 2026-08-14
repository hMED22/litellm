"""Tests for per-second transcription cost calculation."""

import json
from pathlib import Path

import pytest

import litellm
from litellm.llms.openai.cost_calculation import cost_per_second

REPO_ROOT = Path(__file__).parents[4]
PRICE_FILES = (
    REPO_ROOT / "model_prices_and_context_window.json",
    REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json",
)


def _register_stt(name: str, **pricing: float) -> None:
    litellm.register_model(
        {
            name: {
                "mode": "audio_transcription",
                "litellm_provider": "openai",
                **pricing,
            }
        },
        persist_across_reloads=False,
    )


def _transcription_models(price_file: Path) -> dict:
    return {
        name: info
        for name, info in json.loads(price_file.read_text()).items()
        if isinstance(info, dict) and info.get("mode") == "audio_transcription"
    }


def test_input_rate_bills_when_output_rate_is_zero():
    """A declared-but-zero output rate must not suppress the real input rate."""
    _register_stt(
        "test-stt-zero-output",
        input_cost_per_second=5e-05,
        output_cost_per_second=0.0,
    )

    prompt_cost, completion_cost = cost_per_second(
        model="test-stt-zero-output", custom_llm_provider="openai", duration=300.0
    )

    assert prompt_cost == pytest.approx(0.015)
    assert completion_cost == 0.0


def test_input_and_output_rates_are_billed_independently():
    _register_stt(
        "test-stt-both-rates",
        input_cost_per_second=1e-04,
        output_cost_per_second=2e-04,
    )

    prompt_cost, completion_cost = cost_per_second(
        model="test-stt-both-rates", custom_llm_provider="openai", duration=10.0
    )

    assert prompt_cost == pytest.approx(1e-03)
    assert completion_cost == pytest.approx(2e-03)


def test_output_rate_alone_still_bills():
    _register_stt("test-stt-output-only", output_cost_per_second=3e-05)

    prompt_cost, completion_cost = cost_per_second(
        model="test-stt-output-only", custom_llm_provider="openai", duration=60.0
    )

    assert prompt_cost == 0.0
    assert completion_cost == pytest.approx(1.8e-03)


@pytest.mark.parametrize(
    "model, provider",
    [
        ("deepgram/nova-3", "deepgram"),
        ("groq/whisper-large-v3", "groq"),
        ("elevenlabs/scribe_v1", "elevenlabs"),
        ("assemblyai/best", "assemblyai"),
        ("whisper-1", "openai"),
    ],
)
def test_shipped_per_second_models_bill_a_non_zero_cost(model, provider):
    prompt_cost, completion_cost = cost_per_second(model=model, custom_llm_provider=provider, duration=60.0)

    assert prompt_cost + completion_cost > 0.0


@pytest.mark.parametrize("price_file", PRICE_FILES, ids=lambda p: p.name)
def test_no_transcription_model_declares_two_billable_per_second_rates(price_file):
    """Both per-second rates are summed, so declaring the same rate twice double-bills."""
    duplicated = {
        name
        for name, info in _transcription_models(price_file).items()
        if (info.get("input_cost_per_second") or 0) > 0 and (info.get("output_cost_per_second") or 0) > 0
    }

    assert duplicated == set()
