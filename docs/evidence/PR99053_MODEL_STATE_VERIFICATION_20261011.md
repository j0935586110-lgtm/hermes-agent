# Hermes model identity / temporary-model projection: independent review receipt (2026-10-11)

Scope: independent, isolated **fake-provider** verification in NousResearch/hermes-agent. This is evidence for PR [#99053](https://github.com/NousResearch/hermes-agent/pull/99053), complementary to [#98703](https://github.com/NousResearch/hermes-agent/pull/98703). Neither a claim of maintainer approval nor a production billing incident.

## Findings

1. **Upstream main (without #99053): a reproducible split in effective model identity.** At pinned main commit `97a9ae8ee55d31bebce49f1fb6b5ef775323976a`, a legitimate `llm_request` middleware changes only `request.model` from **A** to **B**. A real `AIAgent.run_conversation()` execution with an in-process fake wire records:
   - effective provider wire request model: **B**;
   - `pre_api_request.model`: **A** (although its `request.body.model` correctly says **B**);
   - Relay `model_name` metadata: **A**;
   - `ContextCompressor.model`: **A**; model-context metadata fixture retains an A-sized **256K** limit even though B requires **128K**.

   [Standalone pytest witness](../../evals/test_middleware_model_identity_witness.py) asserts this observed divergence. A successful test means *the divergence was reproduced*, not that the behavior is correct. This tests an allowed middleware rewrite but does not by itself settle the intended API contract.

2. **#99053's scoped pre-build approach aligns state.** At pinned original PR commit `d4fb4a7d08397b2b863dca67d21b15380f903951`, in-process E2E checks saw effective wire/hook/Relay model **B** and the B context limit (**128K**) during the turn; both agent and compressor restore to A after it. Six independent comparison assertions pass on Linux at the original PR HEAD and the local fixed candidate. This demonstrates a workable approach but does **not** prove that the full PR (rather than a smaller generic model projection primitive) is required.

3. **The PR's SQLite transient projection issue was real and fixed in the updated branch.** The old public PR head `d4fb4a7d` changed durable per-session compression protection during a transient model projection; three regression scenarios fail there: write-on-projection, outage during compensating restore, and explicit JSON `null` versus absent key. The fixed commit `8a7d4b60c24b3f1129843ecd8998ab9993f97f75` prevents transient projection writes via a context-scoped compressor identity guard, without disabling genuine compressor/fallback persistence. Eleven SQLite adversarial scenarios pass after the fix, and the old snapshot/compensating-write machinery was removed.

## Reproduction (safe local fake clients)

Run in an isolated worktree / venv. **Never run exploratory test probes against your production `HOME` / `HERMES_HOME`**, because plugin bootstrap may change local settings.

```bash
# Pin a clean copy of upstream main to 97a9ae8ee55d31bebce49f1fb6b5ef775323976a
# Copy evals/test_middleware_model_identity_witness.py into that checkout.
scratch="$(mktemp -d)"
mkdir -p "$scratch/.hermes"
HOME="$scratch" HERMES_HOME="$scratch/.hermes" PYTHONPATH="$PWD" \
  python -m pytest -q -s -p no:cacheprovider evals/test_middleware_model_identity_witness.py
```

The test uses in-process fake OpenAI clients; no live model API or real credentials are required. Ensure the checkout has Hermes's test dependencies installed. The current evidence branch is for diagnostic reproduction; it does not change upstream main or the public routing contract.

For #99053 on its current PR branch:
```bash
python -m pytest -q -p no:cacheprovider \
  tests/agent/test_runtime_override.py \
  tests/e2e/test_runtime_override_e2e.py \
  tests/agent/test_runtime_override_redteam_sqlite.py
python scripts/check --base origin/main
```

## Validation matrix (all against pinned source)

| Test | Linux x86_64 | macOS arm64 |
| --- | --- | --- |
| main mismatch witness, isolated fake wire | **1/1 PASS** (reproduces divergent identities) | not run |
| PR original head: unit + E2E | passed | **44/44 PASS** |
| PR original head: additional model identity/context probes | **6/6 PASS** | **2/2 PASS** |
| fixed candidate: unit + E2E + SQLite suite | **55/55 PASS** | **57/57 PASS**, including 2 identity probes |
| fixed candidate: expanded compression/fallback/concurrency regressions | **107/107 PASS** | not run |
| fix integrated with latest main `97a9ae8` | **107/107 PASS** plus **6/6** probes | not separately run |
| post-merge local `scripts/check --base origin/main` | **11/11 PASS** | not run |
| hosted GitHub workflows | `action_required` (maintainer approval required) | same |

The candidate patch transferred to macOS had SHA-256:
`e3ac4710606ac7547c35bf64e441b4b5779a185f879f500db1038ba3e0f8b766`.
Corresponding Linux/Mac `runtime_override.py` from original head SHA-256:
`62cd734d9380b8b4e07f82905d01dab349791866367aea22c849fa0c39817140`.

Current public PR HEAD after synchronizing upstream main: `11b57f34b8c1ce9885ae89db3d726105cd1c3fed`. Mergeability and CI are distinct: cleanly mergeable does not imply accepted or passing hosted workflows.

## Architectural conclusion / review request

- **#98703:** pre-agent *turn-entry* model/provider/API-mode route; Hermes resolves endpoint and credentials through trusted configuration.
- **#99053:** conditional, same-provider, model-only *post-context/pre-request-assembly* override; align model-dependent context/cache/reasoning state, fallback handoff, and restore on exit.
- **existing request/execution middleware:** provider request shaping, observation, interception; rewriting a payload `model` does not automatically project Agent-owned model state.

The user need is real and the feature is relevant beyond any one consuming project (including J1 Agent's optional Hermes backend). Still open for maintainer decision: should Hermes expose a **smaller shared model-state projection/observability primitive** usable by plugins, or retain a narrowly scoped `pre_llm_call` model-only hook? The evidence above proves a need for consistent state, not exclusivity of this one implementation.

Limitations: no live-provider billing check; no claimed full project test suite, no hosted CI pass or official maintainer approval. None of the original user profile configuration files or credentials are published.
