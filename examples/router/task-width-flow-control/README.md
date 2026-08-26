<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Task-Width Flow Control

This standalone crate demonstrates Dynamo's experimental `FlowControlPolicy` interface. It treats the session ID in each request as a task key and allows a fixed number of requests from that task to remain active at once.

`classify` stays pending when a task has reached its width. A `Completed` or `Aborted` lifecycle event releases capacity and wakes pending classifiers. If the bounded event mailbox coalesces events, `Reconcile` supplies the router's live-request snapshot so the policy can remove stale reservations. Requests without both a request ID and session metadata pass through without creating policy state.

Dynamo continues to own request ordering, worker eligibility, placement, reservations, and accounting. This policy controls only when a request enters the router's ordering stage.

For the capacity-only ThunderAgent reduction, `classify` remains the default pass-through hook because Order already waits for downstream capacity. llm-d sticky routing remains a Place-only policy and does not require a custom Classify hook. The task-width policy in this crate demonstrates the TAPER-style case that does require Classify deferral.

## Use the Policy

Construct the policy and pass its configuration to `KvScheduler::start_with_flow_control` in the process that owns the router:

```rust
use std::num::NonZeroUsize;

use dynamo_kv_router::scheduling::FlowControlConfig;
use dynamo_task_width_flow_control_example::TaskWidthPolicy;

let flow_control = FlowControlConfig::new(TaskWidthPolicy::new(NonZeroUsize::MIN));
```

The normal `KvScheduler::start` path does not create or invoke a flow-control policy.

## Test the Example

From the Dynamo repository root, run:

```bash
cargo test -p dynamo-task-width-flow-control-example
```
