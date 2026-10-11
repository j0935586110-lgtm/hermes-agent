"""Runtime override support for ``pre_llm_call`` plugin hooks.

A plugin may return ``{"runtime_override": {...}}`` from ``pre_llm_call`` to
proactively override the LLM API call parameters for the current turn:

    {"context": "recalled text...",            # existing behavior, unchanged
     "runtime_override": {
         "model": "gpt-5.6",
     }}

Contract (mirrors the ``pre_failover_decision`` redirect contract):

* ``redirect`` is *error-driven*: it is applied by the retry/failover machinery
  only after an API call has failed.  ``runtime_override`` is *proactive*: it is
  applied before the first API call of the turn.  The two do not conflict —
  redirect rewrites identity on the failover path, runtime_override rewrites
  identity on the primary path.
* The override is ephemeral and turn-scoped: it lives on ``agent._runtime_override``,
  is re-resolved on every turn prologue, is never persisted to the session DB and
  is never injected into the user message / session history.
* Unsupported keys are logged with a one-line warning and ignored (never crash).
* The override switches the model only: ``provider``, ``api_mode``, ``api_key``
  and ``base_url`` are intentionally unsupported (an earlier contract allowed
  them; they were removed) — credentials never flow through the hook return and
  the endpoint/wire are resolved only from the provider's existing settings, so
  a plugin cannot pick a network destination or a different wire for the
  session.
* TRUST IMPLICATION: a model switch stays inside the existing provider route —
  it never touches credentials or the endpoint, so it cannot redirect the
  session elsewhere.  Installing a plugin therefore grants it this power; only
  install plugins you trust.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Keys a plugin may override.  Anything else is logged and ignored.
#: ``system_prompt`` is intentionally NOT supported: it is the prompt-cache
#: prefix (byte-stable for the life of a conversation), so overriding it would
#: invalidate the cache and drop the core instructions. Model routing does not
#: need it; persona switching needs a separate cache-safe design.
RUNTIME_OVERRIDE_KEYS = frozenset({"model"})

#: All supported keys are plain non-empty strings.
_STRING_KEYS = frozenset({"model"})

#: Identity keys whose change makes the effective route different from the
#: pre-override route.  A route change refreshes derived route state the
#: same way ``switch_model`` / ``_try_activate_fallback`` do.
_ROUTE_KEYS = frozenset({"model"})


def validate_runtime_override(overrides: Any) -> dict[str, str]:
    """Type-check + filter a plugin-provided ``runtime_override`` dict.

    Returns only the supported, correctly-typed keys.  Unsupported keys and
    wrong-typed values are logged with a one-line warning and dropped, so a
    misbehaving plugin can never crash the turn.
    """
    if not isinstance(overrides, dict):
        logger.warning(
            "pre_llm_call runtime_override ignored: expected dict, got %s",
            type(overrides).__name__,
        )
        return {}
    valid: dict[str, str] = {}
    for key, value in overrides.items():
        if key not in RUNTIME_OVERRIDE_KEYS:
            logger.warning(
                "pre_llm_call runtime_override: unsupported key %r ignored "
                "(supported: %s)",
                key,
                ", ".join(sorted(RUNTIME_OVERRIDE_KEYS)),
            )
            continue
        if key in _STRING_KEYS and (not isinstance(value, str) or not value.strip()):
            logger.warning(
                "pre_llm_call runtime_override: key %r must be a non-empty "
                "string, ignored",
                key,
            )
            continue
        valid[key] = value.strip() if isinstance(value, str) else value
    return valid


def _refresh_derived_route_state(agent: Any, overrides: dict[str, str]) -> None:
    """Refresh provider-derived state for the overridden route.

    Mirrors what the canonical route switch (``switch_model`` /
    ``_try_activate_fallback``) does when the active route changes: the
    switched-to provider's ``request_overrides`` (``extra_body``) replaces the
    previous provider's, and ``runtime_capabilities`` is re-resolved for the
    new model/endpoint.  The model-owned state the canonical switch projects
    (prompt-cache flags, context compressor, reasoning config) is refreshed by
    ``_project_override_model_state`` — the activation step this calls.  All of
    it is best-effort — a resolution failure must never crash the turn; the
    scope snapshot still restores the original values on exit.
    """
    try:
        from agent.agent_runtime_helpers import (
            _apply_switched_provider_request_overrides,
        )

        _apply_switched_provider_request_overrides(
            agent, str(overrides.get("provider") or agent.provider)
        )
    except Exception as _ro_exc:
        logger.warning(
            "runtime_override: request_overrides refresh failed (%s); "
            "keeping previous value for the scope",
            _ro_exc,
            exc_info=True,
        )
    try:
        from agent.native_compaction import resolve_native_compaction_capabilities

        agent.runtime_capabilities = resolve_native_compaction_capabilities(
            model=getattr(agent, "model", "") or "",
            base_url=getattr(agent, "base_url", "") or "",
            provider=getattr(agent, "provider", "") or "",
            is_codex_backend=(
                (getattr(agent, "provider", "") or "").strip().lower()
                == "openai-codex"
            ),
        )
    except Exception as _cap_exc:
        logger.warning(
            "runtime_override: runtime_capabilities refresh failed (%s); "
            "keeping previous value for the scope",
            _cap_exc,
            exc_info=True,
        )
    _project_override_model_state(agent, overrides)


def _isolated_context_compressor(compressor: Any) -> Any:
    """Scope-owned copy of ``compressor`` the canonical projection may re-point.

    The canonical model-owned projection re-points ``agent.context_compressor``
    through ``update_model``, which ASSIGNS the model-owned fields (model,
    context length, thresholds, calibration state) on the instance — a shallow
    copy isolates those assignments from the session compressor.  Never mutate
    the pre-override compressor in place: its reference is restored on scope
    exit, so an in-place re-point would leave the override's context length on
    the session compressor.  ``update_model`` only assigns scalars (no
    nested-container mutation), so a shallow copy fully isolates the
    projection.  The copy keeps the durable session handles: a compression that
    actually fires inside the scope (or a fallback that supersedes it) is a
    real session event and its bookkeeping must persist.
    """
    import copy as _copy

    return _copy.copy(compressor)


class _ModelProjectionFailed(RuntimeError):
    """The model-owned projection could not be completed.

    Raised instead of silently keeping a half-projected agent: the override owns
    the route only if the whole projection lands.  The scope catches it and fails
    open to the configured route.
    """


def _project_override_model_state(agent: Any, overrides: dict[str, str]) -> None:
    """Point the model-owned derived state at the overridden model (P1-1).

    Reuses the canonical model-owned projection from ``switch_model``
    (``_apply_model_owned_state`` in agent_runtime_helpers) instead of
    re-implementing it: prompt-cache flags, the per-model context-length
    re-point of ``agent.context_compressor``, and ``reasoning_config`` are
    projected exactly as a real switch projects them, so the request/preflight
    path (which reads ``context_compressor.threshold_tokens``,
    ``_use_prompt_caching`` and ``reasoning_config``) never sees the pre-override
    model's values while ``agent.model`` is the override model.

    The canonical projection re-points ``agent.context_compressor`` in place, so
    a scope-owned copy is swapped in first (``_isolated_context_compressor``);
    the pre-override reference is already in the scope snapshot (it is one of
    ``_DERIVED_ATTRS``) and is restored on exit.  All-or-nothing: a resolution
    failure raises ``_ModelProjectionFailed`` so the scope fails open to the
    configured route — a half-projected agent would serve one request that mixes
    two models' context length, cache policy and reasoning state.
    """
    _cc = getattr(agent, "context_compressor", None)
    if _cc is None:
        # Minimal/bare agent: it carries none of the model-owned state this
        # projection owns (no compressor to re-point, no derived context length),
        # so there is no second model's truth to contradict — nothing to project.
        return
    try:
        agent.context_compressor = _isolated_context_compressor(_cc)
    except Exception as _iso_exc:
        # Never project onto the session compressor in place, and never stand on
        # a half-applied route: fail open to the configured route.
        logger.warning(
            "runtime_override: compressor isolation failed; the override is "
            "not applied for this turn",
            exc_info=True,
        )
        raise _ModelProjectionFailed("compressor isolation failed") from _iso_exc
    try:
        from agent.agent_runtime_helpers import _apply_model_owned_state

        # Model-only projection uses the canonical update_model() reset logic,
        # but those resets are ephemeral.  Suppress writes only in the current
        # execution context and only for this scope-owned compressor.  Unlike
        # temporarily clearing _session_db, another thread never sees broken
        # session binding or loses permission to persist a real session event.
        from agent.ephemeral_model_projection import suppress_ephemeral_projection_writes

        with suppress_ephemeral_projection_writes(agent.context_compressor):
            _apply_model_owned_state(
                agent,
                str(overrides.get("model") or getattr(agent, "model", "") or ""),
                snapshot=None,  # the scope owns in-memory route rollback
            )
        agent._cached_system_prompt = None
        # The cached system prompt's context-file caps are scaled by the model's
        # context window (build_system_prompt_parts reads
        # context_compressor.context_length). Invalidate so the next build
        # re-scales them — mirrors switch_model's `_cached_system_prompt = None`.
        # Snapshot/restore via _DERIVED_ATTRS makes this a no-leak transaction.
    except Exception as _moe_exc:
        # The projection may have partially mutated the copy AND the agent's
        # model-owned attributes: go back to the session compressor and let the
        # scope roll the whole transaction back (fail open to the configured
        # route) rather than serve a request that mixes two models' state.
        agent.context_compressor = _cc
        logger.warning(
            "runtime_override: model-owned state projection failed; the override "
            "is not applied for this turn",
            exc_info=True,
        )
        raise _ModelProjectionFailed("model-owned state projection failed") from _moe_exc


class _RuntimeOverrideScope:
    """Context manager that temporarily applies an override to ``agent``.

    The override is an atomic route transaction, not a second, narrower route
    mutation primitive: it snapshots and restores every route-owned datum the
    canonical switch (``switch_model`` / ``_try_activate_fallback``) manages,
    so the effective route stays consistent through request construction,
    request middleware, ``pre_api_request``, wire execution, and response
    handling.

    Precedence with the error-driven failover path: a proactive override owns
    only the primary attempt.  If ``_try_activate_fallback`` succeeds while the
    scope is open, the fallback supersedes the override.  Supersession is an
    EXPLICIT handoff, never inferred: the fallback call site invokes
    ``consume_runtime_override(agent)``, which finds this scope (registered as
    ``agent._active_runtime_override_scope`` in ``__enter__``) and calls
    ``supersede()``.  ``__exit__`` then sees the ``_superseded`` flag and skips
    the route-identity restore (which would clobber the freshly activated
    fallback); the caller has already cleared ``agent._runtime_override`` so
    retries stay on the fallback route.
    """

    _ATTRS = ("model",)
    _MISSING = object()
    # Per-turn route marker used for usage attribution while an ephemeral
    # model override is live (see agent/turn_usage.py).
    _TURN_ROUTE_TASK = "runtime_override"
    # Route-owned derived state the wire path reads and the route refresh
    # (``_refresh_derived_route_state``) rewrites for the new model.
    # Snapshot/restore the set unconditionally so untouched fields are no-ops
    # and changed fields revert atomically on exit.
    #
    # P1-1: this includes the MODEL-OWNED state the canonical switch
    # (``switch_model``) projects and request/preflight code reads, so the scope
    # never leaves two models' truths in play:
    #   * ``context_compressor`` is an OBJECT — the snapshot holds the reference,
    #     activation swaps in a scope-owned copy (see
    #     ``_project_override_model_state``), and exit restores the reference.
    #     The pre-override compressor is never mutated in place, so the
    #     override's context length cannot leak into the session compressor.
    #   * ``reasoning_config`` / ``_use_prompt_caching`` /
    #     ``_use_native_cache_layout`` are plain values — snapshot/restore as-is.
    #   * ``_config_context_length`` and ``_custom_providers`` are written by the
    #     shared projection (``_resolve_switch_context_length`` / the custom-provider
    #     refresh); restoring them keeps the override a no-leak transaction.
    _DERIVED_ATTRS = (
        "request_overrides",
        "runtime_capabilities",
        "context_compressor",
        "reasoning_config",
        "_use_prompt_caching",
        "_use_native_cache_layout",
        "_config_context_length",
        "_custom_providers",
        "_cached_system_prompt",
    )

    def __init__(self, agent: Any, overrides: dict[str, str]) -> None:
        self.agent = agent
        self.overrides = overrides
        self._snapshot: dict[str, Any] = {}
        self._client_kwargs_snapshot: Optional[dict[str, Any]] = None
        self._transport_cache_snapshot: Optional[dict[str, Any]] = None
        # The ephemeral override must not become the session's recorded route:
        # mark the turn as a temporary route for usage attribution.
        self._turn_route_task_snapshot: Any = None
        self._turn_route_task_set = False
        # Projection writes are suppressed only for its own execution context.
        # Unlike compensation-on-exit, this has no DB snapshot or lost-update window.
        self._superseded = False
        # Set once a restore ran (a failed projection aborts inside __enter__),
        # so the real __exit__ cannot restore a second time.
        self._ended = False
        # Set when restoring a route-owned value failed.  A failed override may
        # only fall open when the rollback actually completed — a request issued
        # against a partially restored route is exactly the mixed state this
        # scope must never produce.
        self._restore_failed = False
        # Nesting: the registered outermost scope is this scope's parent, and a
        # supersession propagates to every live child (a child restoring its own
        # snapshot would otherwise resurrect the route the fallback replaced).
        self._parent_scope: Optional[_RuntimeOverrideScope] = None
        self._children: list[_RuntimeOverrideScope] = []

    def __enter__(self) -> _RuntimeOverrideScope:
        agent = self.agent
        # ── Snapshot phase ─────────────────────────────────────────────
        # NOTE: agent._runtime_override is the canonical source.
        # TurnContext.runtime_override is derived from it at construction
        # time.  The call path in turn_api_call.py reads only the agent
        # attribute; keep that as the single point of truth.
        ov = self.overrides
        for name in self._ATTRS:
            if name in ov:
                self._snapshot[name] = getattr(agent, name, self._MISSING)
        for name in self._DERIVED_ATTRS:
            self._snapshot[name] = getattr(agent, name, self._MISSING)
        # request_overrides is replaced wholesale on activation, never mutated
        # in place — a shallow copy is enough to make the restore exact.
        if isinstance(self._snapshot.get("request_overrides"), dict):
            self._snapshot["request_overrides"] = dict(
                self._snapshot["request_overrides"]
            )
        # _client_kwargs feeds the per-request OpenAI-wire client.  Snapshot a
        # shallow copy so in-place mutation is reversible.
        ck = getattr(agent, "_client_kwargs", None)
        if isinstance(ck, dict):
            self._client_kwargs_snapshot = dict(ck)
            self._snapshot["_client_kwargs"] = ck
        # Transport cache: snapshot the content so exit can restore the
        # pre-override cache instead of leaving an override-mode transport in
        # the agent's per-mode cache.
        _tc = getattr(agent, "_transport_cache", self._MISSING)
        if isinstance(_tc, dict):
            self._transport_cache_snapshot = dict(_tc)
            self._snapshot["_transport_cache"] = _tc

        # Temporary per-turn route: keeps `sessions.model` on the session route
        # while the override model's tokens still land in session_model_usage.
        # An existing route task (e.g. voice_chat) owns the marker and is left alone.
        if not getattr(agent, "_turn_route_task", ""):
            self._turn_route_task_snapshot = getattr(agent, "_turn_route_task", "")
            agent._turn_route_task = self._TURN_ROUTE_TASK
            self._turn_route_task_set = True

        # ── Activation phase ───────────────────────────────────────────
        # Activation failure skips automatic exit, so roll back the in-memory
        # route explicitly.  Temporary model projection never clears DB rows.
        try:
            for name in self._ATTRS:
                if name in ov:
                    # Normalize before storing: the emptiness check below strips,
                    # but the stored value must be stripped too or " gpt-5.6 "
                    # flows into agent.model and onto the wire.
                    val = str(ov[name]).strip()
                    if not val:
                        logger.warning("runtime_override: empty value for %r ignored", name)
                        # Drop the rejected key so the route-refresh check below
                        # does not act on a value that was just declared ignored.
                        # In the canonical flow ov is a validated dict (never
                        # holds empties); this only guards direct callers.
                        ov.pop(name, None)
                        continue
                    ov[name] = val
                    setattr(agent, name, ov[name])
            if _ROUTE_KEYS.intersection(ov):
                _refresh_derived_route_state(agent, ov)
        except _ModelProjectionFailed:
            # Fail open to the configured route: the override is NOT applied, so
            # the request runs on the route the agent already had — never on a
            # half-projected mixture of the two.
            logger.warning(
                "runtime_override: override not applied for this turn "
                "(model projection could not be completed); the configured route stands",
                exc_info=True,
            )
            self._abort()
            return self
        except BaseException:
            logger.warning(
                "runtime_override: activation failed; rolling the scope back",
                exc_info=True,
            )
            try:
                self.__exit__(*sys.exc_info())
            except Exception:
                logger.warning(
                    "runtime_override: rollback after activation failure failed",
                    exc_info=True,
                )
            raise

        # Register as the agent's active scope so the fallback handoff
        # (consume_runtime_override) can find and supersede this scope.
        # First-registered-wins: a nested scope (middleware/retry re-entry)
        # must not steal the registration from the outermost attempt scope —
        # the fallback sites all run outside it, so superseding the outermost
        # scope is always correct.  An inner scope therefore registers only
        # when no scope is registered, and unregisters only when it holds the
        # registration.
        if getattr(agent, "_active_runtime_override_scope", None) is None:
            agent._active_runtime_override_scope = self
        else:
            # Nested scope (middleware/retry re-entry): remember the parent so a
            # fallback handoff supersedes this scope too — a superseded parent
            # must not leave a live child free to restore the override route.
            self._parent_scope = agent._active_runtime_override_scope
            self._parent_scope._children.append(self)
        return self

    def _restore_turn_route_task(self) -> None:
        """Drop the temporary route marker so later turns are attributed normally."""
        if not self._turn_route_task_set:
            return
        self._turn_route_task_set = False
        try:
            setattr(self.agent, "_turn_route_task", self._turn_route_task_snapshot or "")
        except Exception:  # health: allow BLE001 -- restore must not mask the turn's outcome
            self._restore_failed = True
            logger.warning(
                "runtime_override: could not restore _turn_route_task", exc_info=True
            )

    def _abort(self) -> None:
        """Recover the in-memory route after a failed model projection.

        The projection suppresses temporary durable writes; no DB compensation
        is needed. Complete rollback safely falls open to the configured route.
        Failed in-memory rollback blocks the request and clears the override.
        """
        self.agent._runtime_override = {}
        try:
            self.__exit__(None, None, None)
        except Exception:
            self._restore_failed = True
            logger.warning(
                "runtime_override: rollback after a failed override failed",
                exc_info=True,
            )
        if self._restore_failed:
            raise _ModelProjectionFailed(
                "override rollback could not be completed; refusing to issue a "
                "request against a partially restored route"
            )

    def supersede(self) -> None:
        """Explicitly hand the route to the fallback chain.

        Marks the scope superseded so ``__exit__`` skips the route-identity
        restore (which would clobber the freshly activated fallback), and
        clears ``agent._runtime_override`` so no retry iteration re-applies
        the failed override.  Called by ``consume_runtime_override`` from the
        ``_try_activate_fallback`` success sites.

        Propagates to every live nested scope: the fallback sites all run
        outside the attempt scopes, so an inner scope still open at that moment
        would otherwise restore its own snapshot — the override route — over the
        fallback identity on its way out.
        """
        self._superseded = True
        self.agent._runtime_override = {}
        for child in list(self._children):
            child.supersede()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        agent = self.agent
        try:
            if self._superseded:
                # The fallback chain took ownership of the route mid-scope and
                # rebuilt every route-owned datum for the fallback route.
                # Restoring the snapshotted pre-override state here would
                # clobber the fallback — intentional: the fallback path owns
                # the route now.  The caller has already cleared
                # ``agent._runtime_override`` via ``consume_runtime_override``,
                # so retries stay on the fallback route.
                return
            if self._ended:
                # A failed projection already restored this scope inside
                # __enter__; restoring twice is not harmless (it would rewrite
                # values the turn has moved on from).
                return
            self._ended = True
            for name, value in self._snapshot.items():
                if name in ("_client_kwargs", "_transport_cache"):
                    continue  # restored from the snapshots below
                if value is self._MISSING:
                    # Some intentionally minimal agents never had this attr and
                    # projection didn't manufacture it.  Absence is already
                    # restored; trying delattr on it is not a rollback failure.
                    if not hasattr(agent, name):
                        continue
                    try:
                        delattr(agent, name)
                    except Exception:
                        self._restore_failed = True
                        logger.warning(
                            "runtime_override: could not remove %s during restore",
                            name,
                            exc_info=True,
                        )
                    continue
                try:
                    setattr(agent, name, value)
                except Exception:
                    self._restore_failed = True
                    logger.warning(
                        "runtime_override: could not restore %s", name, exc_info=True
                    )
            if self._client_kwargs_snapshot is not None:
                ck = getattr(agent, "_client_kwargs", None)
                if isinstance(ck, dict):
                    ck.clear()
                    ck.update(self._client_kwargs_snapshot)
                else:
                    self._restore_failed = True
                    logger.warning(
                        "runtime_override: _client_kwargs is no longer a dict; the "
                        "pre-override client state could not be restored"
                    )
            if self._transport_cache_snapshot is not None:
                tc = getattr(agent, "_transport_cache", None)
                if isinstance(tc, dict):
                    tc.clear()
                    tc.update(self._transport_cache_snapshot)
                else:
                    self._restore_failed = True
                    logger.warning(
                        "runtime_override: _transport_cache is no longer a dict; the "
                        "pre-override transport cache could not be restored"
                    )
        finally:
            # The temporary route marker is dropped on EVERY exit path, including the
            # fallback handoff (`_superseded`) where the route itself is left to the
            # fallback: a leaked marker would mis-attribute every later turn.
            self._restore_turn_route_task()
            # Unregister on every exit path (normal restore AND superseded
            # skip), and only when this scope holds the registration.
            if self._parent_scope is not None:
                try:
                    self._parent_scope._children.remove(self)
                except ValueError:
                    logger.debug(
                        "runtime_override: nested scope was not registered with its parent"
                    )
            if getattr(agent, "_active_runtime_override_scope", None) is self:
                agent._active_runtime_override_scope = None
        if self._restore_failed:
            agent._runtime_override = {}
            if exc_type is None:
                raise _ModelProjectionFailed(
                    "runtime override route restore incomplete; refusing to report success"
                )
            # Preserve the original exception; it already fails this attempt.
            logger.error("runtime override route restore incomplete during failed turn")


def apply_runtime_override(agent: Any, overrides: dict[str, str]) -> _RuntimeOverrideScope:
    """Return a context manager that applies ``overrides`` to ``agent``."""
    return _RuntimeOverrideScope(agent, overrides)


def consume_runtime_override(agent: Any) -> None:
    """Explicit supersede handoff: the fallback chain took ownership of the route.

    A proactive override owns only the primary attempt: once
    ``_try_activate_fallback`` succeeds, the fallback route supersedes it for
    the remainder of the logical request, so the next retry iteration must not
    re-enter the route that just failed.  Every ``_try_activate_fallback``
    success site calls this on success.

    When an override scope is active (``agent._active_runtime_override_scope``),
    this marks it superseded and clears ``agent._runtime_override``; when no
    scope is active (exception-driven fallbacks run after the attempt's scope
    already restored the agent) it clears the turn-scoped override directly so
    the current request stays on the fallback route.  None-safe: a bare agent
    without the registration attribute never raises.
    """
    scope = getattr(agent, "_active_runtime_override_scope", None)
    if scope is not None:
        scope.supersede()
        return
    try:
        agent._runtime_override = {}
    except AttributeError:
        pass  # bare test agent without the attribute
