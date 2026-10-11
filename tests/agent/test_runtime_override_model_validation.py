"""Regression contract for PR #99053 finding G1: an override model must be checked with
the same validator the `/model` switch uses, and only a positively unrecognized id is
dropped.

The PR body promises that a name which cannot be recognized for the active provider is
logged and the override dropped, so a plugin typo cannot fail an otherwise valid turn.
Custom/self-hosted endpoints and an unavailable catalog must keep working (fail open).
"""

from __future__ import annotations

from agent.runtime_override import apply_runtime_override
from tests.agent.test_runtime_override import (
    _OVERRIDE_MODEL,
    _SESSION_MODEL,
    _build_model_owned_agent,
    _patch_model_owned_resolution,
)

_VALIDATOR = "hermes_cli.models_validate.validate_requested_model"


def _agent(monkeypatch):
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    agent._cached_system_prompt = "FROZEN_SYSTEM_PROMPT_BYTES"
    return agent


def test_unrecognized_model_override_is_dropped(monkeypatch):
    monkeypatch.setattr(
        _VALIDATOR,
        lambda model, provider, **kw: {
            "accepted": False, "persist": False, "recognized": False,
            "message": "not a known model",
        },
    )
    agent = _agent(monkeypatch)

    with apply_runtime_override(agent, {"model": "definitely-not-a-model-xyz"}):
        inside_model = agent.model
        inside_prompt = agent._cached_system_prompt

    assert inside_model == _SESSION_MODEL, "an unrecognized model must not reach the wire"
    assert inside_prompt == "FROZEN_SYSTEM_PROMPT_BYTES", (
        "a dropped override must not invalidate the cached system prompt"
    )


def test_recognized_model_override_applies(monkeypatch):
    monkeypatch.setattr(
        _VALIDATOR,
        lambda model, provider, **kw: {
            "accepted": True, "persist": True, "recognized": True, "message": "",
        },
    )
    agent = _agent(monkeypatch)

    with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
        assert agent.model == _OVERRIDE_MODEL


def test_validator_failure_fails_open(monkeypatch):
    """An offline/unknown catalog must never turn a legitimate override into a failure."""
    def _boom(model, provider, **kw):
        raise RuntimeError("catalog offline")

    monkeypatch.setattr(_VALIDATOR, _boom)
    agent = _agent(monkeypatch)

    with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
        assert agent.model == _OVERRIDE_MODEL


def test_custom_endpoint_model_is_not_dropped(monkeypatch):
    """Custom endpoints report recognized=False but are still acceptable."""
    monkeypatch.setattr(
        _VALIDATOR,
        lambda model, provider, **kw: {
            "accepted": True, "persist": False, "recognized": False, "message": "",
        },
    )
    agent = _agent(monkeypatch)

    with apply_runtime_override(agent, {"model": "my-self-hosted-model"}):
        assert agent.model == "my-self-hosted-model"
