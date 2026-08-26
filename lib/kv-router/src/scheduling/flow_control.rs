// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::FutureExt;
use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, watch};
use tokio_util::sync::CancellationToken;

use super::SessionContext;
use super::types::KvSchedulerError;
use crate::protocols::WorkerWithDpRank;

const DEFAULT_MAX_PENDING_CLASSIFICATIONS: usize = 1_024;
const DEFAULT_EVENT_CHANNEL_CAPACITY: usize = 1_024;
const DEFAULT_EVENT_TIMEOUT: Duration = Duration::from_secs(1);

/// Request facts available to a flow-control policy before router ordering.
///
/// The view owns its data so [`FlowControlPolicy::classify`] may remain pending
/// without borrowing scheduler state.
#[derive(Debug)]
pub struct ClassifyRequest {
    request_id: Option<String>,
    policy_class: Option<String>,
    ingress_at: Instant,
    deadline: Option<Instant>,
    raw_isl_tokens: usize,
    cached_tokens: usize,
    scheduling_cost_tokens: usize,
    session_context: Option<SessionContext>,
}

impl ClassifyRequest {
    pub fn new(
        ingress_at: Instant,
        raw_isl_tokens: usize,
        cached_tokens: usize,
        scheduling_cost_tokens: usize,
    ) -> Self {
        Self {
            request_id: None,
            policy_class: None,
            ingress_at,
            deadline: None,
            raw_isl_tokens,
            cached_tokens,
            scheduling_cost_tokens,
            session_context: None,
        }
    }

    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    pub fn with_policy_class(mut self, policy_class: impl Into<String>) -> Self {
        self.policy_class = Some(policy_class.into());
        self
    }

    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn with_session_context(mut self, session_context: SessionContext) -> Self {
        self.session_context = Some(session_context);
        self
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn policy_class(&self) -> Option<&str> {
        self.policy_class.as_deref()
    }

    pub fn ingress_at(&self) -> Instant {
        self.ingress_at
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub fn raw_isl_tokens(&self) -> usize {
        self.raw_isl_tokens
    }

    pub fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    pub fn scheduling_cost_tokens(&self) -> usize {
        self.scheduling_cost_tokens
    }

    pub fn session_context(&self) -> Option<&SessionContext> {
        self.session_context.as_ref()
    }
}

/// Per-request scheduling inputs returned to the router-owned ordering stage.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Classification {
    policy_class: Option<String>,
    due_at: Option<Instant>,
    scheduling_cost_tokens: Option<usize>,
}

impl Classification {
    pub fn with_policy_class(mut self, policy_class: impl Into<String>) -> Self {
        self.policy_class = Some(policy_class.into());
        self
    }

    pub fn with_due_at(mut self, due_at: Instant) -> Self {
        self.due_at = Some(due_at);
        self
    }

    pub fn with_scheduling_cost_tokens(mut self, scheduling_cost_tokens: usize) -> Self {
        self.scheduling_cost_tokens = Some(scheduling_cost_tokens);
        self
    }

    pub fn policy_class(&self) -> Option<&str> {
        self.policy_class.as_deref()
    }

    pub fn due_at(&self) -> Option<Instant> {
        self.due_at
    }

    pub fn scheduling_cost_tokens(&self) -> Option<usize> {
        self.scheduling_cost_tokens
    }

    pub(crate) fn into_parts(self) -> (Option<String>, Option<Instant>, Option<usize>) {
        (self.policy_class, self.due_at, self.scheduling_cost_tokens)
    }
}

/// Request lifecycle and reconciliation events delivered outside the scheduler actor.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlowControlEvent {
    Dispatched {
        request_id: String,
        worker: WorkerWithDpRank,
    },
    Completed {
        request_id: String,
    },
    Aborted {
        request_id: String,
    },
    WorkerLoadChanged {
        worker: Option<WorkerWithDpRank>,
    },
    Reconcile {
        /// Router-owned requests that have not reached a terminal state.
        ///
        /// Policies use this snapshot to clean request-local state after the
        /// bounded event mailbox coalesces one or more lifecycle events.
        live_request_ids: Arc<[String]>,
    },
}

impl FlowControlEvent {
    fn kind(&self) -> &'static str {
        match self {
            Self::Dispatched { .. } => "dispatched",
            Self::Completed { .. } => "completed",
            Self::Aborted { .. } => "aborted",
            Self::WorkerLoadChanged { .. } => "worker_load_changed",
            Self::Reconcile { .. } => "reconcile",
        }
    }
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct FlowControlPolicyError {
    message: String,
}

/// Router-side limits for one configured flow-control policy.
pub struct FlowControlConfig {
    policy: Arc<dyn FlowControlPolicy>,
    max_pending_classifications: usize,
    event_channel_capacity: usize,
    event_timeout: Duration,
}

impl FlowControlConfig {
    pub fn new(policy: impl FlowControlPolicy) -> Self {
        Self {
            policy: Arc::new(policy),
            max_pending_classifications: DEFAULT_MAX_PENDING_CLASSIFICATIONS,
            event_channel_capacity: DEFAULT_EVENT_CHANNEL_CAPACITY,
            event_timeout: DEFAULT_EVENT_TIMEOUT,
        }
    }

    pub fn with_max_pending_classifications(
        mut self,
        max_pending_classifications: NonZeroUsize,
    ) -> Self {
        self.max_pending_classifications = max_pending_classifications.get();
        self
    }

    pub fn with_event_channel_capacity(mut self, event_channel_capacity: NonZeroUsize) -> Self {
        self.event_channel_capacity = event_channel_capacity.get();
        self
    }

    pub fn with_event_timeout(mut self, event_timeout: Duration) -> Self {
        self.event_timeout = event_timeout;
        self
    }
}

impl FlowControlPolicyError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Experimental request flow-control policy.
///
/// Calls to [`Self::classify`] may run concurrently and remain pending. The
/// router validates the returned classification before entering Order.
#[async_trait]
pub trait FlowControlPolicy: Send + Sync + 'static {
    async fn init(&self) -> Result<(), FlowControlPolicyError> {
        Ok(())
    }

    async fn classify(
        &self,
        _request: ClassifyRequest,
    ) -> Result<Classification, FlowControlPolicyError> {
        Ok(Classification::default())
    }

    async fn on_event(&self, _event: FlowControlEvent) -> Result<(), FlowControlPolicyError> {
        Ok(())
    }

    async fn teardown(&self) -> Result<(), FlowControlPolicyError> {
        Ok(())
    }
}

pub(crate) struct FlowControlRuntime {
    policy: Arc<dyn FlowControlPolicy>,
    pending_classifications: Arc<Semaphore>,
    active_classifications: Arc<AtomicUsize>,
    classifications_idle: Arc<tokio::sync::Notify>,
    max_pending_classifications: usize,
    event_tx: mpsc::Sender<FlowControlEvent>,
    live_requests: Arc<Mutex<HashSet<String>>>,
    reconcile_pending: Arc<AtomicBool>,
    reconcile_notify: Arc<tokio::sync::Notify>,
    shutdown: CancellationToken,
    shutdown_complete: watch::Receiver<bool>,
}

struct ClassificationPermit {
    _permit: OwnedSemaphorePermit,
    active_classifications: Arc<AtomicUsize>,
    classifications_idle: Arc<tokio::sync::Notify>,
}

impl Drop for ClassificationPermit {
    fn drop(&mut self) {
        if self.active_classifications.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.classifications_idle.notify_one();
        }
    }
}

impl FlowControlRuntime {
    pub(crate) async fn initialize(
        config: FlowControlConfig,
        shutdown: CancellationToken,
    ) -> Result<Arc<Self>, KvSchedulerError> {
        let init_policy = Arc::clone(&config.policy);
        let init = AssertUnwindSafe(async move { init_policy.init().await }).catch_unwind();
        tokio::pin!(init);
        tokio::select! {
            result = &mut init => map_policy_future_result("init", result)?,
            _ = shutdown.cancelled() => return Err(KvSchedulerError::SubscriberShutdown),
        }

        let max_pending_classifications = config.max_pending_classifications;
        let (event_tx, event_rx) = mpsc::channel(config.event_channel_capacity);
        let active_classifications = Arc::new(AtomicUsize::new(0));
        let classifications_idle = Arc::new(tokio::sync::Notify::new());
        let reconcile_pending = Arc::new(AtomicBool::new(false));
        let reconcile_notify = Arc::new(tokio::sync::Notify::new());
        let live_requests = Arc::new(Mutex::new(HashSet::new()));
        let (shutdown_complete_tx, shutdown_complete) = watch::channel(false);
        let pending_classifications = Arc::new(Semaphore::new(max_pending_classifications));
        let runtime = Arc::new(Self {
            policy: Arc::clone(&config.policy),
            pending_classifications: Arc::clone(&pending_classifications),
            active_classifications: Arc::clone(&active_classifications),
            classifications_idle: Arc::clone(&classifications_idle),
            max_pending_classifications,
            event_tx,
            live_requests: Arc::clone(&live_requests),
            reconcile_pending: Arc::clone(&reconcile_pending),
            reconcile_notify: Arc::clone(&reconcile_notify),
            shutdown: shutdown.clone(),
            shutdown_complete,
        });

        tokio::spawn(async move {
            run_event_pump(
                config.policy,
                event_rx,
                live_requests,
                reconcile_pending,
                reconcile_notify,
                pending_classifications,
                active_classifications,
                classifications_idle,
                config.event_timeout,
                shutdown,
            )
            .await;
            let _ = shutdown_complete_tx.send(true);
        });

        Ok(runtime)
    }

    pub(crate) async fn classify(
        &self,
        request: ClassifyRequest,
    ) -> Result<Classification, KvSchedulerError> {
        let deadline = request.deadline();
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            return Err(KvSchedulerError::DeadlineExceeded);
        }
        let permit = self.acquire_classification_permit()?;
        let policy = Arc::clone(&self.policy);
        let classify = AssertUnwindSafe(async move {
            let _permit = permit;
            policy.classify(request).await
        })
        .catch_unwind();
        tokio::pin!(classify);
        if let Some(deadline) = deadline {
            tokio::select! {
                result = &mut classify => {
                    map_policy_future_result("classify", result)
                }
                _ = self.shutdown.cancelled() => Err(KvSchedulerError::SubscriberShutdown),
                _ = tokio::time::sleep_until(deadline.into()) => Err(KvSchedulerError::DeadlineExceeded),
            }
        } else {
            tokio::select! {
                result = &mut classify => {
                    map_policy_future_result("classify", result)
                }
                _ = self.shutdown.cancelled() => Err(KvSchedulerError::SubscriberShutdown),
            }
        }
    }

    fn acquire_classification_permit(&self) -> Result<ClassificationPermit, KvSchedulerError> {
        if self.shutdown.is_cancelled() {
            return Err(KvSchedulerError::SubscriberShutdown);
        }
        let permit = Arc::clone(&self.pending_classifications)
            .try_acquire_owned()
            .map_err(|_| {
                if self.shutdown.is_cancelled() || self.pending_classifications.is_closed() {
                    KvSchedulerError::SubscriberShutdown
                } else {
                    KvSchedulerError::FlowControlPendingLimit {
                        limit: self.max_pending_classifications,
                    }
                }
            })?;
        self.active_classifications.fetch_add(1, Ordering::AcqRel);
        Ok(ClassificationPermit {
            _permit: permit,
            active_classifications: Arc::clone(&self.active_classifications),
            classifications_idle: Arc::clone(&self.classifications_idle),
        })
    }

    pub(crate) fn emit(&self, event: FlowControlEvent) {
        if self.shutdown.is_cancelled() {
            return;
        }
        match self.event_tx.try_send(event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => self.request_reconcile(),
            Err(mpsc::error::TrySendError::Closed(_)) => {}
        }
    }

    pub(crate) fn begin_request(&self, request_id: &str) -> bool {
        self.live_requests.lock().insert(request_id.to_owned())
    }

    pub(crate) fn finish_request(&self, event: FlowControlEvent) {
        let request_id = match &event {
            FlowControlEvent::Completed { request_id, .. }
            | FlowControlEvent::Aborted { request_id } => request_id,
            _ => {
                debug_assert!(false, "finish_request requires a terminal event");
                return;
            }
        };
        self.live_requests.lock().remove(request_id);
        self.emit(event);
    }

    fn request_reconcile(&self) {
        request_reconcile(
            self.reconcile_pending.as_ref(),
            self.reconcile_notify.as_ref(),
        );
    }

    pub(crate) async fn wait_for_shutdown(&self) {
        let mut shutdown_complete = self.shutdown_complete.clone();
        while !*shutdown_complete.borrow_and_update() {
            if shutdown_complete.changed().await.is_err() {
                return;
            }
        }
    }

    pub(crate) fn pending_classification_count(&self) -> usize {
        self.active_classifications.load(Ordering::Relaxed)
    }
}

fn map_policy_future_result<T>(
    operation: &str,
    result: Result<Result<T, FlowControlPolicyError>, Box<dyn std::any::Any + Send + 'static>>,
) -> Result<T, KvSchedulerError> {
    match result {
        Ok(result) => result.map_err(KvSchedulerError::FlowControlPolicy),
        Err(payload) => Err(KvSchedulerError::FlowControlPolicy(
            FlowControlPolicyError::new(format!(
                "{operation} panicked: {}",
                panic_message(payload.as_ref())
            )),
        )),
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

#[allow(clippy::too_many_arguments)]
async fn run_event_pump(
    policy: Arc<dyn FlowControlPolicy>,
    mut event_rx: mpsc::Receiver<FlowControlEvent>,
    live_requests: Arc<Mutex<HashSet<String>>>,
    reconcile_pending: Arc<AtomicBool>,
    reconcile_notify: Arc<tokio::sync::Notify>,
    pending_classifications: Arc<Semaphore>,
    active_classifications: Arc<AtomicUsize>,
    classifications_idle: Arc<tokio::sync::Notify>,
    event_timeout: Duration,
    shutdown: CancellationToken,
) {
    let mut event_failure_warned = false;
    loop {
        let event = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            _ = reconcile_notify.notified() => {
                if !reconcile_pending.swap(false, Ordering::AcqRel) {
                    continue;
                }
                let mut live_request_ids = live_requests.lock().iter().cloned().collect::<Vec<_>>();
                live_request_ids.sort_unstable();
                FlowControlEvent::Reconcile {
                    live_request_ids: live_request_ids.into(),
                }
            }
            event = event_rx.recv() => {
                let Some(event) = event else {
                    break;
                };
                event
            }
        };
        let event_kind = event.kind();
        let event_delivery = AssertUnwindSafe(policy.on_event(event)).catch_unwind();
        tokio::pin!(event_delivery);
        let event_sleep = tokio::time::sleep(event_timeout);
        tokio::pin!(event_sleep);
        let result = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            result = &mut event_delivery => Some(result),
            _ = &mut event_sleep => None,
        };
        let failed = match result {
            Some(Ok(Ok(()))) => false,
            Some(Ok(Err(error))) => {
                log_event_failure(&mut event_failure_warned, event_kind, &error.to_string());
                true
            }
            Some(Err(payload)) => {
                log_event_failure(
                    &mut event_failure_warned,
                    event_kind,
                    panic_message(payload.as_ref()),
                );
                true
            }
            None => {
                log_event_failure(&mut event_failure_warned, event_kind, "timeout");
                true
            }
        };
        if failed && event_kind != "reconcile" {
            request_reconcile(reconcile_pending.as_ref(), reconcile_notify.as_ref());
        }
    }

    pending_classifications.close();
    wait_for_classifications(&active_classifications, &classifications_idle).await;
    match AssertUnwindSafe(policy.teardown()).catch_unwind().await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => tracing::warn!(%error, "Flow-control policy teardown failed"),
        Err(payload) => tracing::warn!(
            reason = panic_message(payload.as_ref()),
            "Flow-control policy teardown panicked"
        ),
    }
}

fn request_reconcile(reconcile_pending: &AtomicBool, reconcile_notify: &tokio::sync::Notify) {
    if !reconcile_pending.swap(true, Ordering::AcqRel) {
        reconcile_notify.notify_one();
    }
}

fn log_event_failure(warned: &mut bool, event_kind: &str, reason: &str) {
    if *warned {
        tracing::debug!(
            event_kind,
            reason,
            "Flow-control policy event delivery failed"
        );
    } else {
        tracing::warn!(
            event_kind,
            reason,
            "Flow-control policy event delivery failed"
        );
        *warned = true;
    }
}

async fn wait_for_classifications(
    active_classifications: &AtomicUsize,
    classifications_idle: &tokio::sync::Notify,
) {
    loop {
        let idle = classifications_idle.notified();
        if active_classifications.load(Ordering::Acquire) == 0 {
            return;
        }
        idle.await;
    }
}

#[cfg(test)]
mod tests {
    use std::future;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use tokio::sync::Notify;

    use super::*;

    struct PassThrough;

    #[async_trait]
    impl FlowControlPolicy for PassThrough {}

    #[tokio::test]
    async fn default_policy_passes_through_without_overrides() {
        let request = ClassifyRequest::new(Instant::now(), 128, 32, 96)
            .with_request_id("request-1")
            .with_policy_class("latency");

        let result = PassThrough.classify(request).await.unwrap();

        assert_eq!(result, Classification::default());
    }

    struct PendingPolicy {
        initialized: Arc<AtomicBool>,
        entered: Arc<Notify>,
        released: Arc<Notify>,
    }

    #[async_trait]
    impl FlowControlPolicy for PendingPolicy {
        async fn init(&self) -> Result<(), FlowControlPolicyError> {
            self.initialized.store(true, Ordering::Release);
            Ok(())
        }

        async fn classify(
            &self,
            _request: ClassifyRequest,
        ) -> Result<Classification, FlowControlPolicyError> {
            assert!(self.initialized.load(Ordering::Acquire));
            self.entered.notify_one();
            self.released.notified().await;
            Ok(Classification::default())
        }
    }

    #[tokio::test]
    async fn runtime_initializes_before_classification_and_bounds_pending_calls() {
        let initialized = Arc::new(AtomicBool::new(false));
        let entered = Arc::new(Notify::new());
        let released = Arc::new(Notify::new());
        let config = FlowControlConfig::new(PendingPolicy {
            initialized: Arc::clone(&initialized),
            entered: Arc::clone(&entered),
            released: Arc::clone(&released),
        })
        .with_max_pending_classifications(NonZeroUsize::new(1).unwrap());
        let shutdown = CancellationToken::new();
        let runtime = FlowControlRuntime::initialize(config, shutdown.clone())
            .await
            .unwrap();
        assert!(initialized.load(Ordering::Acquire));

        let first_runtime = Arc::clone(&runtime);
        let first = tokio::spawn(async move {
            first_runtime
                .classify(ClassifyRequest::new(Instant::now(), 1, 0, 1))
                .await
        });
        entered.notified().await;
        assert_eq!(runtime.pending_classification_count(), 1);

        let second = runtime
            .classify(ClassifyRequest::new(Instant::now(), 1, 0, 1))
            .await;
        assert!(matches!(
            second,
            Err(KvSchedulerError::FlowControlPendingLimit { limit: 1 })
        ));

        released.notify_one();
        first.await.unwrap().unwrap();
        assert_eq!(runtime.pending_classification_count(), 0);
        shutdown.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn runtime_expires_a_pending_classification_at_the_authoritative_deadline() {
        let config = FlowControlConfig::new(PendingPolicy {
            initialized: Arc::new(AtomicBool::new(false)),
            entered: Arc::new(Notify::new()),
            released: Arc::new(Notify::new()),
        });
        let runtime = FlowControlRuntime::initialize(config, CancellationToken::new())
            .await
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        let request = ClassifyRequest::new(Instant::now(), 1, 0, 1).with_deadline(deadline);

        let result = runtime.classify(request).await;

        assert!(matches!(result, Err(KvSchedulerError::DeadlineExceeded)));
    }

    struct BlockingEventPolicy {
        first: AtomicBool,
        entered: Arc<Notify>,
        release: Arc<Notify>,
        events: mpsc::UnboundedSender<&'static str>,
        reconciliations: mpsc::UnboundedSender<Arc<[String]>>,
    }

    #[async_trait]
    impl FlowControlPolicy for BlockingEventPolicy {
        async fn on_event(&self, event: FlowControlEvent) -> Result<(), FlowControlPolicyError> {
            self.events.send(event.kind()).unwrap();
            if let FlowControlEvent::Reconcile { live_request_ids } = event {
                self.reconciliations.send(live_request_ids).unwrap();
            }
            if self.first.swap(false, Ordering::AcqRel) {
                self.entered.notify_one();
                self.release.notified().await;
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn full_event_mailbox_coalesces_a_lost_event_into_reconcile() {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let (reconciliation_tx, mut reconciliation_rx) = mpsc::unbounded_channel();
        let config = FlowControlConfig::new(BlockingEventPolicy {
            first: AtomicBool::new(true),
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
            events: event_tx,
            reconciliations: reconciliation_tx,
        })
        .with_event_channel_capacity(NonZeroUsize::new(1).unwrap());
        let shutdown = CancellationToken::new();
        let runtime = FlowControlRuntime::initialize(config, shutdown.clone())
            .await
            .unwrap();

        assert!(runtime.begin_request("live"));
        assert!(runtime.begin_request("mailbox-full"));
        runtime.finish_request(FlowControlEvent::Completed {
            request_id: "first".to_string(),
        });
        entered.notified().await;
        assert_eq!(event_rx.recv().await, Some("completed"));
        runtime.emit(FlowControlEvent::Completed {
            request_id: "second".to_string(),
        });
        runtime.finish_request(FlowControlEvent::Aborted {
            request_id: "mailbox-full".to_string(),
        });
        release.notify_one();

        tokio::time::timeout(Duration::from_millis(250), async {
            loop {
                if event_rx.recv().await == Some("reconcile") {
                    break;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(
            reconciliation_rx.recv().await.as_deref(),
            Some(["live".to_string()].as_slice())
        );
        shutdown.cancel();
        runtime.wait_for_shutdown().await;
    }

    struct ShutdownPolicy {
        entered: Arc<Notify>,
        classification_dropped: Arc<AtomicBool>,
        torn_down: Arc<AtomicBool>,
    }

    struct DropSignal(Arc<AtomicBool>);

    impl Drop for DropSignal {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    #[async_trait]
    impl FlowControlPolicy for ShutdownPolicy {
        async fn classify(
            &self,
            _request: ClassifyRequest,
        ) -> Result<Classification, FlowControlPolicyError> {
            let _drop_signal = DropSignal(Arc::clone(&self.classification_dropped));
            self.entered.notify_one();
            future::pending().await
        }

        async fn teardown(&self) -> Result<(), FlowControlPolicyError> {
            assert!(self.classification_dropped.load(Ordering::Acquire));
            self.torn_down.store(true, Ordering::Release);
            Ok(())
        }
    }

    #[tokio::test]
    async fn shutdown_cancels_classification_before_awaiting_teardown() {
        let entered = Arc::new(Notify::new());
        let classification_dropped = Arc::new(AtomicBool::new(false));
        let torn_down = Arc::new(AtomicBool::new(false));
        let shutdown = CancellationToken::new();
        let runtime = FlowControlRuntime::initialize(
            FlowControlConfig::new(ShutdownPolicy {
                entered: Arc::clone(&entered),
                classification_dropped: Arc::clone(&classification_dropped),
                torn_down: Arc::clone(&torn_down),
            }),
            shutdown.clone(),
        )
        .await
        .unwrap();
        let classify_runtime = Arc::clone(&runtime);
        let classify = tokio::spawn(async move {
            classify_runtime
                .classify(ClassifyRequest::new(Instant::now(), 1, 0, 1))
                .await
        });
        entered.notified().await;

        shutdown.cancel();
        runtime.wait_for_shutdown().await;

        assert!(matches!(
            classify.await.unwrap(),
            Err(KvSchedulerError::SubscriberShutdown)
        ));
        assert!(classification_dropped.load(Ordering::Acquire));
        assert!(torn_down.load(Ordering::Acquire));
    }

    struct SlowEventPolicy {
        completed: mpsc::UnboundedSender<()>,
    }

    #[async_trait]
    impl FlowControlPolicy for SlowEventPolicy {
        async fn on_event(&self, event: FlowControlEvent) -> Result<(), FlowControlPolicyError> {
            if matches!(event, FlowControlEvent::Reconcile { .. }) {
                future::pending().await
            } else {
                self.completed.send(()).unwrap();
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn timed_out_event_does_not_stall_later_delivery() {
        let (completed_tx, mut completed_rx) = mpsc::unbounded_channel();
        let shutdown = CancellationToken::new();
        let runtime = FlowControlRuntime::initialize(
            FlowControlConfig::new(SlowEventPolicy {
                completed: completed_tx,
            })
            .with_event_timeout(Duration::from_millis(5)),
            shutdown.clone(),
        )
        .await
        .unwrap();
        runtime.request_reconcile();
        runtime.emit(FlowControlEvent::Completed {
            request_id: "after-timeout".to_string(),
        });

        tokio::time::timeout(Duration::from_millis(250), completed_rx.recv())
            .await
            .unwrap();
        shutdown.cancel();
        runtime.wait_for_shutdown().await;
    }

    struct TimedOutTerminalPolicy {
        reconciliations: mpsc::UnboundedSender<Arc<[String]>>,
    }

    #[async_trait]
    impl FlowControlPolicy for TimedOutTerminalPolicy {
        async fn on_event(&self, event: FlowControlEvent) -> Result<(), FlowControlPolicyError> {
            match event {
                FlowControlEvent::Completed { .. } => future::pending().await,
                FlowControlEvent::Reconcile { live_request_ids } => {
                    self.reconciliations.send(live_request_ids).unwrap();
                    Ok(())
                }
                _ => Ok(()),
            }
        }
    }

    #[tokio::test]
    async fn failed_terminal_event_triggers_authoritative_reconciliation() {
        let (reconciliation_tx, mut reconciliation_rx) = mpsc::unbounded_channel();
        let shutdown = CancellationToken::new();
        let runtime = FlowControlRuntime::initialize(
            FlowControlConfig::new(TimedOutTerminalPolicy {
                reconciliations: reconciliation_tx,
            })
            .with_event_timeout(Duration::from_millis(5)),
            shutdown.clone(),
        )
        .await
        .unwrap();
        assert!(runtime.begin_request("completed"));

        runtime.finish_request(FlowControlEvent::Completed {
            request_id: "completed".to_string(),
        });

        let live_request_ids =
            tokio::time::timeout(Duration::from_millis(250), reconciliation_rx.recv())
                .await
                .unwrap()
                .unwrap();
        assert!(live_request_ids.is_empty());
        shutdown.cancel();
        runtime.wait_for_shutdown().await;
    }

    struct PanickingClassifier;

    #[async_trait]
    impl FlowControlPolicy for PanickingClassifier {
        async fn classify(
            &self,
            _request: ClassifyRequest,
        ) -> Result<Classification, FlowControlPolicyError> {
            panic!("classifier panic")
        }
    }

    #[tokio::test]
    async fn classifier_panic_is_isolated_as_a_policy_failure() {
        let shutdown = CancellationToken::new();
        let runtime = FlowControlRuntime::initialize(
            FlowControlConfig::new(PanickingClassifier),
            shutdown.clone(),
        )
        .await
        .unwrap();

        let result = runtime
            .classify(ClassifyRequest::new(Instant::now(), 1, 0, 1))
            .await;

        assert!(matches!(
            result,
            Err(KvSchedulerError::FlowControlPolicy(_))
        ));
        shutdown.cancel();
        runtime.wait_for_shutdown().await;
    }

    struct FailingClassifier;

    #[async_trait]
    impl FlowControlPolicy for FailingClassifier {
        async fn classify(
            &self,
            _request: ClassifyRequest,
        ) -> Result<Classification, FlowControlPolicyError> {
            Err(FlowControlPolicyError::new("classification failed"))
        }
    }

    #[tokio::test]
    async fn classifier_error_remains_distinct_from_overload() {
        let shutdown = CancellationToken::new();
        let runtime = FlowControlRuntime::initialize(
            FlowControlConfig::new(FailingClassifier),
            shutdown.clone(),
        )
        .await
        .unwrap();

        let result = runtime
            .classify(ClassifyRequest::new(Instant::now(), 1, 0, 1))
            .await;

        let Err(KvSchedulerError::FlowControlPolicy(error)) = result else {
            panic!("expected a flow-control policy error");
        };
        assert_eq!(error.to_string(), "classification failed");
        shutdown.cancel();
        runtime.wait_for_shutdown().await;
    }
}
