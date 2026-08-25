// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use super::SessionContext;
use super::types::KvSchedulerError;
use crate::protocols::WorkerWithDpRank;

const DEFAULT_MAX_PENDING_CLASSIFICATIONS: usize = 1_024;

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
        context_tokens: Option<usize>,
    },
    Aborted {
        request_id: String,
    },
    Reconcile,
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
}

impl FlowControlConfig {
    pub fn new(policy: impl FlowControlPolicy) -> Self {
        Self {
            policy: Arc::new(policy),
            max_pending_classifications: DEFAULT_MAX_PENDING_CLASSIFICATIONS,
        }
    }

    pub fn with_max_pending_classifications(
        mut self,
        max_pending_classifications: NonZeroUsize,
    ) -> Self {
        self.max_pending_classifications = max_pending_classifications.get();
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
    max_pending_classifications: usize,
    shutdown: CancellationToken,
}

impl FlowControlRuntime {
    pub(crate) async fn initialize(
        config: FlowControlConfig,
        shutdown: CancellationToken,
    ) -> Result<Arc<Self>, KvSchedulerError> {
        tokio::select! {
            result = config.policy.init() => result.map_err(KvSchedulerError::FlowControlPolicy)?,
            _ = shutdown.cancelled() => return Err(KvSchedulerError::SubscriberShutdown),
        }

        let max_pending_classifications = config.max_pending_classifications;
        let runtime = Arc::new(Self {
            policy: config.policy,
            pending_classifications: Arc::new(Semaphore::new(max_pending_classifications)),
            max_pending_classifications,
            shutdown: shutdown.clone(),
        });

        let policy = Arc::clone(&runtime.policy);
        tokio::spawn(async move {
            shutdown.cancelled().await;
            if let Err(error) = policy.teardown().await {
                tracing::warn!(%error, "Flow-control policy teardown failed");
            }
        });

        Ok(runtime)
    }

    pub(crate) async fn classify(
        &self,
        request: ClassifyRequest,
    ) -> Result<Classification, KvSchedulerError> {
        let _permit = self.acquire_classification_permit()?;
        let deadline = request.deadline();
        if let Some(deadline) = deadline {
            if deadline <= Instant::now() {
                return Err(KvSchedulerError::DeadlineExceeded);
            }
            tokio::select! {
                result = self.policy.classify(request) => {
                    result.map_err(KvSchedulerError::FlowControlPolicy)
                }
                _ = self.shutdown.cancelled() => Err(KvSchedulerError::SubscriberShutdown),
                _ = tokio::time::sleep_until(deadline.into()) => Err(KvSchedulerError::DeadlineExceeded),
            }
        } else {
            tokio::select! {
                result = self.policy.classify(request) => {
                    result.map_err(KvSchedulerError::FlowControlPolicy)
                }
                _ = self.shutdown.cancelled() => Err(KvSchedulerError::SubscriberShutdown),
            }
        }
    }

    fn acquire_classification_permit(&self) -> Result<OwnedSemaphorePermit, KvSchedulerError> {
        Arc::clone(&self.pending_classifications)
            .try_acquire_owned()
            .map_err(|_| KvSchedulerError::FlowControlPendingLimit {
                limit: self.max_pending_classifications,
            })
    }
}

#[cfg(test)]
mod tests {
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

        let second = runtime
            .classify(ClassifyRequest::new(Instant::now(), 1, 0, 1))
            .await;
        assert!(matches!(
            second,
            Err(KvSchedulerError::FlowControlPendingLimit { limit: 1 })
        ));

        released.notify_one();
        first.await.unwrap().unwrap();
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
}
