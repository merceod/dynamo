// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use async_trait::async_trait;

use super::SessionContext;
use crate::protocols::WorkerWithDpRank;

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

#[cfg(test)]
mod tests {
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
}
