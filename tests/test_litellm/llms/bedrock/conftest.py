"""Shared fixtures for bedrock tests.

Forces the anthropic_beta_headers_manager to use the local JSON file rather
than fetching from a remote URL. This keeps tests deterministic and aligned
with the local file the team actually edits.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _use_local_anthropic_beta_headers(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")

    from litellm import anthropic_beta_headers_manager

    anthropic_beta_headers_manager._BETA_HEADERS_CONFIG = None

    yield

    anthropic_beta_headers_manager._BETA_HEADERS_CONFIG = None
