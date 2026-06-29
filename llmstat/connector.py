"""LLM API connector layer.

Architecture:
  localai (Docker, default) ──→ fallback ──→ openai (API key)

Both backends implement the same chat-completion interface so the
statistical methodology layer is backend-agnostic.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from llmstat.config import config


class LLMError(Exception):
    """Raised when all LLM backends fail."""


class LLMConnector:
    """Unified LLM interface with automatic fallback.

    Priority: localai → openai
    """

    def __init__(self, timeout: float = 120.0) -> None:
        self._timeout = timeout
        self._backends = _build_backend_chain()

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2) -> str:
        """Send a chat completion request.

        Tries backends in priority order; raises LLMError if all fail.
        """
        errors: list[str] = []
        for backend in self._backends:
            try:
                return _chat_sync(backend, system_prompt, user_prompt, temperature, self._timeout)
            except Exception as exc:
                errors.append(f"{backend['name']}: {exc}")
        raise LLMError("All LLM backends failed:\n" + "\n".join(errors))

    async def chat_async(
        self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2
    ) -> str:
        """Async version of chat()."""
        errors: list[str] = []
        for backend in self._backends:
            try:
                return await _chat_async(backend, system_prompt, user_prompt, temperature, self._timeout)
            except Exception as exc:
                errors.append(f"{backend['name']}: {exc}")
        raise LLMError("All LLM backends failed:\n" + "\n".join(errors))

    def batch_chat(
        self,
        prompts: list[tuple[str, str]],
        *,
        temperature: float = 0.2,
        max_concurrency: int = 20,
    ) -> list[str]:
        """Run multiple chat requests concurrently."""
        return asyncio.run(self.batch_chat_async(prompts, temperature=temperature, max_concurrency=max_concurrency))

    async def batch_chat_async(
        self,
        prompts: list[tuple[str, str]],
        *,
        temperature: float = 0.2,
        max_concurrency: int = 20,
    ) -> list[str]:
        """Async batch with bounded concurrency."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded(system: str, user: str) -> str:
            async with semaphore:
                return await self.chat_async(system, user, temperature=temperature)

        tasks = [_bounded(sys, usr) for sys, usr in prompts]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_backend_chain() -> list[dict[str, Any]]:
    """Build the priority-ordered backend list."""
    chain: list[dict[str, Any]] = []
    if config.backend == "localai":
        chain.append({
            "name": "localai",
            "base_url": config.localai_base_url,
            "model": config.localai_model,
            "api_key": config.localai_api_key,
        })
    # Add openai as fallback regardless of primary backend choice
    if config.openai_api_key and config.backend != "openai":
        chain.append({
            "name": "openai",
            "base_url": config.openai_base_url,
            "model": config.openai_model,
            "api_key": config.openai_api_key,
        })
    elif config.backend == "openai":
        chain.append({
            "name": "openai",
            "base_url": config.openai_base_url,
            "model": config.openai_model,
            "api_key": config.openai_api_key,
        })
    # Always include localai as fallback if it wasn't primary
    if config.backend == "openai":
        chain.append({
            "name": "localai",
            "base_url": config.localai_base_url,
            "model": config.localai_model,
            "api_key": config.localai_api_key,
        })
    return chain


def _build_payload(backend: dict[str, str], system: str, user: str, temperature: float) -> dict[str, Any]:
    return {
        "model": backend["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }


def _chat_sync(
    backend: dict[str, str], system: str, user: str, temperature: float, timeout: float
) -> str:
    """Synchronous chat via httpx."""
    payload = _build_payload(backend, system, user, temperature)
    headers = {
        "Authorization": f"Bearer {backend['api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{backend['base_url'].rstrip('/')}/chat/completions"
    response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


async def _chat_async(
    backend: dict[str, str], system: str, user: str, temperature: float, timeout: float
) -> str:
    """Async chat via httpx."""
    payload = _build_payload(backend, system, user, temperature)
    headers = {
        "Authorization": f"Bearer {backend['api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{backend['base_url'].rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
