# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""
Shared LLM-backend layer for the GPT-based baselines (M3, M4).

All real-traffic calls go through Azure OpenAI.  The required environment
variables are:

* ``AZURE_OPENAI_ENDPOINT``
* ``AZURE_OPENAI_API_KEY``
* ``AZURE_OPENAI_API_VERSION``

In addition, an ``EchoBackend`` is provided for unit tests / smoke runs:
it parses any candidate-list lines (``- CODE: ...``) out of the user
prompt and echoes the first ``k`` of them back as the ``selected`` list,
so the full pipeline (prompt build → "LLM" → parse → KB filter) can be
exercised offline.

A small ``FixedIntervalRateLimiter`` lives here so the M3/M4 drivers can
fan out across a thread pool while still respecting a global
requests-per-minute ceiling — same primitive the sectionizer uses,
lifted up so we have one implementation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global rate limiter (shared by M3 / M4 thread pools)
# ---------------------------------------------------------------------------
class FixedIntervalRateLimiter:
    """
    Thread-safe global rate limiter using fixed call spacing.

    Example: RPM=250 => interval = 60/250 = 0.24s between call slots.
    Conservative by design — the goal is to never blow past the global
    Azure OpenAI rate cap, even when the worker pool is fully saturated.
    """

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        self._interval_s = 60.0 / float(requests_per_minute)
        self._lock = threading.Lock()
        self._next_allowed_time = time.monotonic()

    def acquire(self) -> None:
        sleep_s = 0.0
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed_time:
                sleep_s = self._next_allowed_time - now
                self._next_allowed_time += self._interval_s
            else:
                self._next_allowed_time = now + self._interval_s
        if sleep_s > 0:
            time.sleep(sleep_s)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def _is_new_style_model(deployment_name: str) -> bool:
    """Heuristic: newer Azure deployments (GPT-5, o-series) use
    ``max_completion_tokens`` instead of ``max_tokens`` and reject any
    non-default ``temperature``.  We key off the deployment-name prefix
    since Azure doesn't otherwise expose the underlying model family at
    call time.
    """
    name = (deployment_name or "").lower()
    return (
        name.startswith("gpt-5")
        or name.startswith("gpt5")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
        # Frontier "chat-latest" deployments (e.g. ``gpt-chat-latest``)
        # follow the GPT-5 contract: ``max_completion_tokens`` only, and
        # they reject any non-default ``temperature``.
        or name.startswith("gpt-chat")
        or "chat-latest" in name
    )


class LLMBackend:
    """Abstract LLM chat backend returning one string response per call."""

    #: Multiplier applied to the token cap each time a call comes back empty
    #: with ``finish_reason == "length"``.
    TRUNCATION_ESCALATION: float = 4.0
    #: Absolute ceiling for that escalation.
    TRUNCATION_CEILING: int = 8192

    def chat(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def chat_n(self, system: str, user: str, n: int) -> List[str]:
        """Return ``n`` independent completions for one prompt.

        The base implementation issues ``n`` sequential :meth:`chat` calls.
        Backends whose server supports the OpenAI ``n`` parameter override
        this to get all ``n`` from a single request, which re-uses one
        prefill instead of ``n`` of them.  Callers must not assume the
        samples are distinct -- at ``temperature=0`` they typically are not.
        """
        return [self.chat(system, user) for _ in range(max(1, int(n)))]


def _first_choice(resp) -> Tuple[str, str]:
    """``(content, finish_reason)`` of choice 0, both normalised to ``str``."""
    choice = resp.choices[0]
    return (choice.message.content or "",
            str(getattr(choice, "finish_reason", "") or ""))


def _all_choices(resp) -> List[str]:
    return [(c.message.content or "") for c in resp.choices]


class AzureOpenAIBackend(LLMBackend):
    """
    Azure OpenAI backend (chat completions in JSON mode).

    Parameters
    ----------
    deployment_name : str
        The Azure deployment name (NOT the underlying model name).  This
        is what gets passed as ``model=`` to ``chat.completions.create``.
    """

    def __init__(
        self,
        deployment_name: str = "gpt-5.3-chat",
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 4,
        truncation_retries: int = 2,
        request_timeout: float = 180.0,
        rate_limiter: Optional[FixedIntervalRateLimiter] = None,
    ) -> None:
        try:
            from openai import AzureOpenAI  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "openai package not installed — `pip install openai` or "
                "use the EchoBackend for offline tests"
            ) from e
        from openai import AzureOpenAI

        self.deployment_name = deployment_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        # Escalation budget for empty finish_reason=length
        # completions, plus the counters the runbook asserts on.
        self.truncation_retries = truncation_retries
        self.n_truncation_retries = 0
        self.n_truncation_lost = 0
        self.rate_limiter = rate_limiter

        # Request-shape state, seeded from the deployment-name heuristic but
        # self-correcting at call time (see ``_maybe_adapt_params``).  New-style
        # models take ``max_completion_tokens`` and reject a custom temperature;
        # legacy GPT-4.x take ``max_tokens`` + ``temperature``.
        new_style = _is_new_style_model(deployment_name)
        self._use_completion_tokens = new_style
        self._send_temperature = not new_style
        # Some deployments enforce Azure's rule that the messages must contain
        # the literal lowercase word "json" before ``response_format`` of type
        # ``json_object`` is accepted.  Every prompt in this repo writes "JSON"
        # in caps, so on those deployments the call 400s.  Same treatment as the
        # two flags above: start on, flip off from the server's own feedback.
        self._send_response_format = True

        self.client = AzureOpenAI(
            api_version=_require_env("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_require_env("AZURE_OPENAI_API_KEY"),
            timeout=request_timeout,
        )

    def _build_kwargs(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        n: int = 1,
    ) -> dict:
        # GPT-5 / o-series / chat-latest: use `max_completion_tokens` and drop
        # `temperature` (only the default is accepted).  Legacy GPT-4.x /
        # GPT-3.5 deployments: keep the old parameters.
        create_kwargs = {
            "model": self.deployment_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._send_response_format:
            create_kwargs["response_format"] = {"type": "json_object"}
        cap = int(max_tokens or self.max_tokens)
        if self._use_completion_tokens:
            create_kwargs["max_completion_tokens"] = cap
        else:
            create_kwargs["max_tokens"] = cap
        if self._send_temperature:
            create_kwargs["temperature"] = self.temperature
        if n > 1:
            create_kwargs["n"] = int(n)
        return create_kwargs

    def _maybe_adapt_params(self, exc: Exception) -> bool:
        """Self-correct the request shape from the server's 400 feedback.

        Azure returns ``unsupported_parameter`` 400s naming the offending
        field (``max_tokens`` → use ``max_completion_tokens``; ``temperature``
        → only the default is accepted; ``response_format`` → the messages must
        contain the word "json").  Rather than key off brittle
        deployment-name prefixes, flip the corresponding flag and let the
        caller retry the *same* note immediately.  Returns ``True`` iff a flag
        actually changed, so the retry loop can avoid burning an attempt /
        backoff sleep.  Each correction is one-way and guarded by its flag, so
        this cannot loop indefinitely.
        """
        msg = str(getattr(exc, "message", "") or exc).lower()
        changed = False
        if (
            not self._use_completion_tokens
            and "max_completion_tokens" in msg
            and "max_tokens" in msg
        ):
            self._use_completion_tokens = True
            changed = True
        if (
            self._send_temperature
            and "temperature" in msg
            and ("not supported" in msg or "unsupported" in msg
                 or "only the default" in msg)
        ):
            self._send_temperature = False
            changed = True
        # "'messages' must contain the word 'json' in some form, to use
        # 'response_format' of type 'json_object'."  The check is on the
        # literal lowercase token, and every prompt in this repo writes "JSON"
        # in caps, so the deployments that enforce it reject the call outright
        # -- four retries, four identical 400s, then a hard failure.  Drop the
        # constraint and let the prompt's own "Return a JSON object"
        # instruction carry it; callers parse defensively.
        if (
            self._send_response_format
            and "response_format" in msg
            and "json" in msg
            and ("must contain" in msg or "in some form" in msg)
        ):
            self._send_response_format = False
            changed = True
        return changed

    def chat(self, system: str, user: str) -> str:
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        last_exc: Optional[Exception] = None
        attempt = 0
        cap = int(self.max_tokens)
        bumps = 0
        while attempt < self.max_retries:
            try:
                resp = self.client.chat.completions.create(
                    **self._build_kwargs(system, user, max_tokens=cap)
                )
                text, reason = _first_choice(resp)
                if not text.strip() and reason == "length":
                    # The cap was spent before a single *content* token was
                    # emitted.  Reasoning models bill reasoning against the
                    # same budget, so a longer answer (e.g. --shortlist-k 10)
                    # can exhaust it while the call still "succeeds": the SDK
                    # raises nothing and the note silently scores zero codes.
                    # Escalate the cap and retry without consuming a failure
                    # attempt.  A response that parsed is untouched, so runs
                    # that never truncated are byte-identical.
                    if bumps < self.truncation_retries:
                        bumps += 1
                        self.n_truncation_retries += 1
                        cap = min(int(cap * self.TRUNCATION_ESCALATION),
                                  self.TRUNCATION_CEILING)
                        LOGGER.warning(
                            "empty completion with finish_reason=length; "
                            "raising the token cap to %d and retrying "
                            "(escalation %d/%d)",
                            cap, bumps, self.truncation_retries,
                        )
                        continue
                    self.n_truncation_lost += 1
                    LOGGER.error(
                        "empty completion with finish_reason=length after "
                        "%d escalation(s) (cap=%d); giving up on this note",
                        bumps, cap,
                    )
                return text
            except Exception as exc:  # pragma: no cover
                # If the server told us exactly which param is wrong, fix the
                # request shape and retry without consuming an attempt.
                if self._maybe_adapt_params(exc):
                    LOGGER.info(
                        "Azure rejected a parameter; adapting "
                        "(completion_tokens=%s, send_temperature=%s) and "
                        "retrying: %s",
                        self._use_completion_tokens,
                        self._send_temperature,
                        exc,
                    )
                    continue
                last_exc = exc
                sleep = min(2.0 ** attempt, 30.0)
                LOGGER.warning(
                    "Azure OpenAI call failed (attempt %d/%d): %s — sleeping %.1fs",
                    attempt + 1, self.max_retries, exc, sleep,
                )
                time.sleep(sleep)
                attempt += 1
        raise RuntimeError(f"Azure OpenAI call failed after retries: {last_exc}")


class LocalOpenAIBackend(LLMBackend):
    """
    OpenAI-compatible backend for a *locally-served open-weight* model.

    Targets any server implementing the OpenAI ``/v1/chat/completions``
    contract — in practice a vLLM ``vllm serve`` endpoint on the local
    2×H100 box, but TGI / SGLang / llama.cpp expose the same API.  It is also
    what an LLM candidate-prior generator should run against, so the candidate
    source stays a model you control rather than an external API.

    Differences from :class:`AzureOpenAIBackend`:

    * ``model`` is the *served* model id (whatever was passed to
      ``vllm serve``, e.g. ``openai/gpt-oss-20b``), not an Azure
      deployment name.
    * the endpoint is a plain ``base_url`` (default
      ``http://localhost:8000/v1``), resolved from ``--base-url`` /
      ``CPT_REC_LOCAL_LLM_BASE_URL``.
    * the API key is a throwaway (servers ignore it, but the OpenAI SDK
      requires a non-empty string).

    Robustness
    ----------
    Not every server / model build supports JSON mode
    (``response_format={"type": "json_object"}``) or the same token
    parameter.  As with the Azure backend we self-correct from the 400
    feedback (:meth:`_maybe_adapt_params`): drop ``response_format`` and
    fall back to the prompt + ``parse_*`` regex; switch ``max_tokens`` →
    ``max_completion_tokens``; drop an unsupported ``temperature``.  Each
    correction is one-way so the retry loop is bounded.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 4,
        truncation_retries: int = 2,
        request_timeout: float = 600.0,
        json_mode: bool = True,
        rate_limiter: Optional[FixedIntervalRateLimiter] = None,
        extra_body: Optional[dict] = None,
    ) -> None:
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "openai package not installed — `pip install openai` "
                "(the same client targets a local vLLM endpoint via base_url)"
            ) from e
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        # Escalation budget for empty finish_reason=length
        # completions, plus the counters the runbook asserts on.
        self.truncation_retries = truncation_retries
        self.n_truncation_retries = 0
        self.n_truncation_lost = 0
        self.rate_limiter = rate_limiter
        self.extra_body = extra_body or None

        # Request-shape flags, self-correcting at call time.  Local servers
        # generally take the *legacy* ``max_tokens`` + ``temperature`` shape,
        # so we start there and only flip on an explicit 400.
        self._use_json_mode = json_mode
        self._use_completion_tokens = False
        self._send_temperature = True
        # Server-side ``n``: one prefill, n sampled continuations.  Disabled
        # permanently for the run the first time an endpoint rejects or
        # ignores it (see ``chat_n``).
        self._use_server_side_n = True

        base_url = (
            base_url
            or os.getenv("CPT_REC_LOCAL_LLM_BASE_URL")
            or "http://localhost:8000/v1"
        )
        api_key = api_key or os.getenv("CPT_REC_LOCAL_LLM_API_KEY") or "EMPTY"
        self.base_url = base_url
        self.client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=request_timeout
        )
        LOGGER.info(
            "LocalOpenAIBackend: model=%s base_url=%s json_mode=%s",
            model, base_url, json_mode,
        )

    def _build_kwargs(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        n: int = 1,
    ) -> dict:
        create_kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._use_json_mode:
            create_kwargs["response_format"] = {"type": "json_object"}
        cap = int(max_tokens or self.max_tokens)
        if self._use_completion_tokens:
            create_kwargs["max_completion_tokens"] = cap
        else:
            create_kwargs["max_tokens"] = cap
        if self._send_temperature:
            create_kwargs["temperature"] = self.temperature
        if n > 1:
            create_kwargs["n"] = int(n)
        if self.extra_body:
            create_kwargs["extra_body"] = self.extra_body
        return create_kwargs

    def _maybe_adapt_params(self, exc: Exception) -> bool:
        """Self-correct request shape from the server's 400 feedback.

        Returns ``True`` iff a flag actually changed so the retry loop can
        avoid burning an attempt.  All corrections are one-way and guarded
        by their flag, so this cannot loop forever.
        """
        msg = str(getattr(exc, "message", "") or exc).lower()
        changed = False
        if self._use_json_mode and "response_format" in msg:
            self._use_json_mode = False
            changed = True
        if (
            not self._use_completion_tokens
            and "max_completion_tokens" in msg
            and "max_tokens" in msg
        ):
            self._use_completion_tokens = True
            changed = True
        if (
            self._send_temperature
            and "temperature" in msg
            and ("not supported" in msg or "unsupported" in msg
                 or "only the default" in msg)
        ):
            self._send_temperature = False
            changed = True
        return changed

    def chat(self, system: str, user: str) -> str:
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        last_exc: Optional[Exception] = None
        attempt = 0
        cap = int(self.max_tokens)
        bumps = 0
        while attempt < self.max_retries:
            try:
                resp = self.client.chat.completions.create(
                    **self._build_kwargs(system, user, max_tokens=cap)
                )
                text, reason = _first_choice(resp)
                if not text.strip() and reason == "length":
                    # The cap was spent before a single *content* token was
                    # emitted.  Reasoning models bill reasoning against the
                    # same budget, so a longer answer (e.g. --shortlist-k 10)
                    # can exhaust it while the call still "succeeds": the SDK
                    # raises nothing and the note silently scores zero codes.
                    # Escalate the cap and retry without consuming a failure
                    # attempt.  A response that parsed is untouched, so runs
                    # that never truncated are byte-identical.
                    if bumps < self.truncation_retries:
                        bumps += 1
                        self.n_truncation_retries += 1
                        cap = min(int(cap * self.TRUNCATION_ESCALATION),
                                  self.TRUNCATION_CEILING)
                        LOGGER.warning(
                            "empty completion with finish_reason=length; "
                            "raising the token cap to %d and retrying "
                            "(escalation %d/%d)",
                            cap, bumps, self.truncation_retries,
                        )
                        continue
                    self.n_truncation_lost += 1
                    LOGGER.error(
                        "empty completion with finish_reason=length after "
                        "%d escalation(s) (cap=%d); giving up on this note",
                        bumps, cap,
                    )
                return text
            except Exception as exc:  # pragma: no cover
                if self._maybe_adapt_params(exc):
                    LOGGER.info(
                        "Local server rejected a parameter; adapting "
                        "(json_mode=%s, completion_tokens=%s, "
                        "send_temperature=%s) and retrying: %s",
                        self._use_json_mode,
                        self._use_completion_tokens,
                        self._send_temperature,
                        exc,
                    )
                    continue
                last_exc = exc
                sleep = min(2.0 ** attempt, 30.0)
                LOGGER.warning(
                    "Local LLM call failed (attempt %d/%d): %s — sleeping %.1fs",
                    attempt + 1, self.max_retries, exc, sleep,
                )
                time.sleep(sleep)
                attempt += 1
        raise RuntimeError(f"Local LLM call failed after retries: {last_exc}")


    def chat_n(self, system: str, user: str, n: int) -> List[str]:
        """
        ``n`` sampled completions from ONE request where the server supports it.

        vLLM samples ``n`` continuations off a single prefill, so on a 27B
        model with a long sectionized prompt this costs a fraction of ``n``
        sequential calls.  Any endpoint that rejects *or silently ignores*
        ``n`` falls back to the sequential base implementation, permanently
        for the rest of the run -- the silent case matters most: one choice
        returned for n samples would make every agreement score 1.0, a
        vacuous ranking that looks perfectly healthy in the output.
        """
        n = max(1, int(n))
        if n == 1 or not self._use_server_side_n:
            return super().chat_n(system, user, n)
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        try:
            resp = self.client.chat.completions.create(
                **self._build_kwargs(system, user, n=n)
            )
        except Exception as exc:  # pragma: no cover - endpoint dependent
            self._use_server_side_n = False
            LOGGER.warning(
                "server rejected n=%d (%s); using %d sequential calls per "
                "note for the rest of the run", n, exc, n,
            )
            return super().chat_n(system, user, n)
        texts = _all_choices(resp)
        if len(texts) < n:
            self._use_server_side_n = False
            LOGGER.warning(
                "server returned %d choice(s) for n=%d -- it is ignoring `n`. "
                "Using sequential calls for the rest of the run.",
                len(texts), n,
            )
            return super().chat_n(system, user, n)
        if not any(t.strip() for t in texts):
            # Every sample empty means the cap was spent before any content
            # token.  The sequential path owns the escalation, so defer this
            # note to it rather than duplicating the ladder here.
            LOGGER.warning(
                "all %d samples empty; re-running this note sequentially so "
                "the token cap can escalate", n,
            )
            return super().chat_n(system, user, n)
        return texts


class EchoBackend(LLMBackend):
    """
    Deterministic stub that does not call any API.

    Looks for candidate-list lines of the form ``- CODE: <desc>`` in the
    user prompt and echoes the first ``k`` of them back as ``selected``.
    If no such lines are found, returns ``{"selected": []}``.

    Useful for offline pipeline tests.
    """

    _CAND_RE = re.compile(r"^- ([A-Z0-9]{5}):", flags=re.MULTILINE)

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def chat(self, system: str, user: str) -> str:
        codes = self._CAND_RE.findall(user)[: self.k]
        return json.dumps({"selected": codes})


def build_backend(name: str, **kwargs) -> LLMBackend:
    name = name.lower()
    if name in ("azure", "azure_openai", "azureopenai"):
        return AzureOpenAIBackend(**kwargs)
    if name in ("local", "vllm", "local_openai", "openai_local"):
        return LocalOpenAIBackend(**kwargs)
    if name == "echo":
        return EchoBackend(**kwargs)
    raise ValueError(f"Unknown LLM backend: {name!r}")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")
_CODE_TOKEN_RE = re.compile(
    r"\b(\d{5}|\d{4}[AFTUM]|[A-Z]\d{4})\b", flags=re.IGNORECASE
)


def parse_selected_codes(response_text: str) -> List[str]:
    """Robustly extract a list of codes from an LLM JSON response."""
    if not response_text:
        return []
    m = _JSON_OBJ_RE.search(response_text)
    if m is not None:
        try:
            obj = json.loads(m.group(0))
            sel = obj.get("selected", [])
            if isinstance(sel, list):
                return [str(c).strip().upper() for c in sel if str(c).strip()]
        except json.JSONDecodeError:
            pass
    # Fallback: scrape anything that looks like a code.
    return [c.upper() for c in _CODE_TOKEN_RE.findall(response_text)]


# ---------------------------------------------------------------------------
# Rich candidate-generation parsing (LLM candidate prior)
# ---------------------------------------------------------------------------
#
# The candidate generator asks the model for a structured shortlist rather
# than a flat ``selected`` list, so the downstream verifier can use per-code
# confidence + the supporting phrase as ``φ(n, c)`` features.  The
# parser is deliberately permissive: it accepts the rich object schema, the
# flat ``selected`` schema (so an EchoBackend / older prompt still works),
# bare code strings inside ``candidate_codes``, and finally a regex scrape.


class CandidateCode:
    """One parsed candidate: code + optional confidence / supporting phrase."""

    __slots__ = ("code", "confidence", "supporting_phrase")

    def __init__(
        self,
        code: str,
        confidence: Optional[float] = None,
        supporting_phrase: Optional[str] = None,
    ) -> None:
        self.code = code.strip().upper()
        self.confidence = confidence
        self.supporting_phrase = supporting_phrase

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CandidateCode({self.code!r}, conf={self.confidence}, "
            f"phrase={self.supporting_phrase!r})"
        )


def _coerce_confidence(value) -> Optional[float]:
    try:
        if value is None:
            return None
        c = float(value)
    except (TypeError, ValueError):
        return None
    # Clamp to [0, 1]; some models emit 0-100 or 0-10 scales.
    if c > 1.0:
        c = c / 100.0 if c > 10.0 else c / 10.0
    return max(0.0, min(1.0, c))


def parse_candidate_codes(response_text: str) -> List[CandidateCode]:
    """Extract ``CandidateCode`` objects from a Mode-1 (direct) response.

    Order is preserved (= the model's own ranking).  Duplicates are kept
    here; de-duplication is the caller's job (so it can merge confidences).
    """
    if not response_text:
        return []
    m = _JSON_OBJ_RE.search(response_text)
    if m is not None:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            raw = obj.get("candidate_codes")
            if raw is None:
                raw = obj.get("selected")  # flat-schema compatibility
            if isinstance(raw, list):
                out: List[CandidateCode] = []
                for item in raw:
                    if isinstance(item, dict):
                        code = item.get("code") or item.get("cpt") or ""
                        if not str(code).strip():
                            continue
                        out.append(
                            CandidateCode(
                                code=str(code),
                                confidence=_coerce_confidence(
                                    item.get("confidence")
                                ),
                                supporting_phrase=(
                                    str(item["supporting_phrase"]).strip()
                                    if item.get("supporting_phrase")
                                    else None
                                ),
                            )
                        )
                    elif str(item).strip():
                        out.append(CandidateCode(code=str(item)))
                if out:
                    return out
    # Fallback: scrape anything that looks like a code, no metadata.
    return [
        CandidateCode(code=c.upper())
        for c in _CODE_TOKEN_RE.findall(response_text)
    ]


def parse_concepts(response_text: str) -> List[str]:
    """Extract procedure-concept search phrases from a Mode-2 response.

    Preferred (rich) schema — one concept fans out into several phrases::

        {"concepts": [
            {"procedure": ..., "anatomy": ..., "approach": ...,
             "search_phrases": ["<standard term>", "<synonym>", ...]},
            ...
        ]}

    All ``search_phrases`` are flattened into one ordered, de-duplicated
    list (concept order, then phrase order) so each is retrieved against
    the KB independently — more surface area for the same procedure.

    Back-compat: also accepts the older single-phrase schema
    ``{"phrase","anatomy","approach"}`` and a flat list of strings under
    ``concepts`` / ``phrases``.  No code parsing here (Mode 2 never emits
    codes).
    """
    if not response_text:
        return []
    m = _JSON_OBJ_RE.search(response_text)
    if m is None:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []
    raw = obj.get("concepts")
    if raw is None:
        raw = obj.get("phrases", [])

    out: List[str] = []
    seen: Set[str] = set()

    def _add(phrase: object) -> None:
        # Collapse internal whitespace; de-dup case-insensitively so
        # repeated phrasings across concepts don't bloat retrieval.
        text = " ".join(str(phrase).split()).strip()
        if not text:
            return
        key = text.lower()
        if key not in seen:
            seen.add(key)
            out.append(text)

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                phrases = item.get("search_phrases")
                if isinstance(phrases, list) and phrases:
                    # Rich schema: one concept -> several search phrases.
                    for p in phrases:
                        if isinstance(p, dict):
                            _add(p.get("phrase", ""))  # tolerate nesting
                        else:
                            _add(p)
                else:
                    # Old single-phrase schema: join the structured fields.
                    parts = [
                        str(item.get(k, "")).strip()
                        for k in ("phrase", "procedure", "anatomy", "approach")
                    ]
                    _add(" ".join(p for p in parts if p))
            else:
                _add(item)
    return out
