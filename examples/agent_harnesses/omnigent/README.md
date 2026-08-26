<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Omnigent through Dynamo

This guide runs one Omnigent task through Codex while a separately deployed Dynamo recipe serves the model. It uses Omnigent's public gateway configuration and agent-file interfaces. It does not deploy Dynamo, prescribe a Kubernetes topology, or require a Dynamo-specific Omnigent patch.

## Deploy Dynamo first

Install the [Dynamo Kubernetes Platform](../../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx), then deploy a supported configuration from the [Dynamo recipe catalog](../../../recipes/README.md). Continue only after the recipe exposes a reachable frontend URL and exact served model.

```bash
export DYNAMO_BASE_URL=http://127.0.0.1:8000
export DYNAMO_MODEL=your-recipe-served-model
curl -fsS "$DYNAMO_BASE_URL/v1/models" | jq -e --arg model "$DYNAMO_MODEL" '.data[]? | select(.id == $model)'
```

## Configure Omnigent

Install a supported Omnigent release and run its setup flow:

```bash
uv tool install --python 3.12 omnigent
omnigent setup
```

Add a Gateway credential for the Codex harness. Use `$DYNAMO_BASE_URL/v1` as the OpenAI-compatible base URL. Supply the recipe's API key when authentication is enabled; use any nonempty value for an unauthenticated local endpoint. Keep credentials in Omnigent's credential store rather than an agent file committed to source control.

Create `dynamo-codex.yaml` outside the target repository:

```yaml
name: dynamo-codex
prompt: Use tools to verify claims before answering.
executor:
  harness: codex
  model: your-recipe-served-model
os_env:
  type: caller_process
  cwd: /absolute/worktree
  sandbox:
    type: auto
```

Replace the model and workspace values. `sandbox.type: auto` selects Omnigent's platform sandbox; do not replace it with `none` on a shared host.

## Run a bounded task

Use a clean, narrowly scoped worktree and request no edits unless they are explicitly authorized:

```bash
omnigent run ./dynamo-codex.yaml \
  --server local \
  --harness codex \
  --model "$DYNAMO_MODEL" \
  -p 'Inspect one named file and report one verified fact. Do not modify files.'
```

Verify that the command returns an assistant response, the target worktree remains unchanged, and Dynamo request traces contain nonempty Codex `thread-id` values normalized as session IDs. Omnigent may create a separate background title request; treat that as a separate session rather than evidence that the main thread persisted across top-level tasks.

## Cleanup

Run this path in a dedicated local environment or container. If cleanup requires `omnigent stop`, remember that the command stops all Omnigent processes owned by that environment; do not use it on a shared host where unrelated Omnigent sessions are running. Never replace scoped cleanup with `pkill`, process-name matching, or another broad host operation.

## Limitations

- This is a stock-Dynamo compatibility path. Omnigent does not currently emit `x-dynamo-session-final`, so it is not ThunderAgent lifecycle-qualified.
- One `omnigent run` invocation does not prove persistent main-thread reuse across multiple top-level prompts.
- The selected Codex harness and Omnigent release own tool permissions, sandbox behavior, credentials, local state, and process cleanup. Revalidate those boundaries after upgrading either dependency.
- Dynamo owns the OpenAI-compatible endpoint, Codex header normalization, request tracing, routing, and inference behavior.
