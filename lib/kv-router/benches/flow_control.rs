// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler overhead with flow control disabled or using a pass-through policy.
//!
//! Run with: `cargo bench -p dynamo-kv-router --bench flow_control`

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike};
use dynamo_kv_router::scheduling::{
    FlowControlConfig, FlowControlPolicy, OverlapSignals, PolicyProfile, ScheduleMode,
    ScheduleRequest,
};
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, DefaultWorkerSelector, LocalScheduler, NoopSequencePublisher,
    RouterQueuePolicy,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

const REQUESTS_PER_BATCH: usize = 128;

type BenchScheduler = LocalScheduler<NoopSequencePublisher, BenchWorkerConfig>;

#[derive(Clone, Default, PartialEq, Eq)]
struct BenchWorkerConfig {
    max_num_batched_tokens: Option<u64>,
    taints: HashSet<String>,
}

impl WorkerConfigLike for BenchWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }

    fn data_parallel_size(&self) -> u32 {
        1
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        None
    }

    fn taints(&self) -> &HashSet<String> {
        &self.taints
    }
}

struct PassThroughPolicy;

impl FlowControlPolicy for PassThroughPolicy {}

async fn make_scheduler(flow_control: bool) -> (Arc<BenchScheduler>, CancellationToken) {
    let workers = HashMap::from([(
        0,
        BenchWorkerConfig {
            max_num_batched_tokens: Some(4_096),
            ..Default::default()
        },
    )]);
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        64,
        HashMap::from([(0, (0, 1))]),
        false,
        0,
        "flow-control-benchmark",
    ));
    let (_workers_tx, workers_rx) = watch::channel(workers);
    let cancellation = CancellationToken::new();
    let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);

    let scheduler = if flow_control {
        LocalScheduler::new_with_policy_profile_and_flow_control(
            slots,
            workers_rx,
            profile,
            64,
            DefaultWorkerSelector::new(None, "flow-control-benchmark"),
            None,
            None,
            None,
            None,
            Duration::from_secs(60),
            true,
            cancellation.clone(),
            "flow-control-benchmark",
            false,
            FlowControlConfig::new(PassThroughPolicy),
        )
        .await
        .unwrap()
    } else {
        LocalScheduler::new_without_overlap_refresh_with_policy_profile(
            slots,
            workers_rx,
            profile,
            64,
            DefaultWorkerSelector::new(None, "flow-control-benchmark"),
            None,
            None,
            None,
            Duration::from_secs(60),
            true,
            cancellation.clone(),
            "flow-control-benchmark",
            false,
        )
        .unwrap()
    };

    (Arc::new(scheduler), cancellation)
}

fn request() -> ScheduleRequest {
    ScheduleRequest {
        mode: ScheduleMode::QueryOnly {
            request_id: Some("flow-control-benchmark".to_owned()),
        },
        deadline: None,
        token_seq: None,
        block_hashes: None,
        isl_tokens: 64,
        lora_name: None,
        expected_output_tokens: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        overlap: OverlapSignals::default(),
        router_hint_candidates: None,
        retain_router_hint_chain: false,
        shared_cache_hits: None,
    }
}

async fn route_batch(scheduler: &BenchScheduler) {
    for _ in 0..REQUESTS_PER_BATCH {
        let response = scheduler.schedule_request(request()).await.unwrap();
        black_box(response.best_worker);
    }
}

fn flow_control(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (disabled, disabled_cancel) = runtime.block_on(make_scheduler(false));
    let (pass_through, pass_through_cancel) = runtime.block_on(make_scheduler(true));

    let mut group = c.benchmark_group("flow_control/schedule_query_only");
    group.throughput(Throughput::Elements(REQUESTS_PER_BATCH as u64));
    group.bench_function("disabled", |b| {
        b.iter(|| runtime.block_on(route_batch(&disabled)));
    });
    group.bench_function("pass_through", |b| {
        b.iter(|| runtime.block_on(route_batch(&pass_through)));
    });
    group.finish();

    disabled_cancel.cancel();
    pass_through_cancel.cancel();
    runtime.block_on(pass_through.wait_for_flow_control_shutdown());
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .noise_threshold(0.03);
    targets = flow_control
}
criterion_main!(benches);
