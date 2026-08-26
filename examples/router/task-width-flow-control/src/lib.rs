// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An experimental flow-control policy that limits active requests per task.
//!
//! The example treats a request's session ID as its task key. Requests without
//! both a request ID and session metadata pass through without policy state.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::{Mutex, MutexGuard};

use async_trait::async_trait;
use dynamo_kv_router::scheduling::{
    Classification, ClassifyRequest, FlowControlEvent, FlowControlPolicy, FlowControlPolicyError,
};
use tokio::sync::watch;

/// Limits the number of classified, non-terminal requests for each task.
pub struct TaskWidthPolicy {
    max_active_requests_per_task: usize,
    state: Mutex<State>,
    state_changed: watch::Sender<u64>,
}

#[derive(Default)]
struct State {
    active_by_task: HashMap<String, usize>,
    task_by_request: HashMap<String, String>,
}

impl TaskWidthPolicy {
    /// Create a policy with a fixed per-task concurrency limit.
    pub fn new(max_active_requests_per_task: NonZeroUsize) -> Self {
        let (state_changed, _) = watch::channel(0);
        Self {
            max_active_requests_per_task: max_active_requests_per_task.get(),
            state: Mutex::new(State::default()),
            state_changed,
        }
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, State>, FlowControlPolicyError> {
        self.state
            .lock()
            .map_err(|_| FlowControlPolicyError::new("task-width policy state lock was poisoned"))
    }

    fn release(&self, request_id: &str) -> Result<bool, FlowControlPolicyError> {
        let mut state = self.lock_state()?;
        Self::release_from_state(&mut state, request_id)
    }

    fn release_from_state(
        state: &mut State,
        request_id: &str,
    ) -> Result<bool, FlowControlPolicyError> {
        let Some(task_id) = state.task_by_request.get(request_id).cloned() else {
            return Ok(false);
        };
        let Some(active) = state.active_by_task.get_mut(&task_id) else {
            return Err(FlowControlPolicyError::new(format!(
                "task-width policy has no active count for request {request_id}"
            )));
        };
        *active = active.checked_sub(1).ok_or_else(|| {
            FlowControlPolicyError::new(format!(
                "task-width policy active count underflow for request {request_id}"
            ))
        })?;
        if *active == 0 {
            state.active_by_task.remove(&task_id);
        }
        state.task_by_request.remove(request_id);
        Ok(true)
    }

    fn reconcile(&self, live_request_ids: &[String]) -> Result<bool, FlowControlPolicyError> {
        let live_request_ids = live_request_ids
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let mut state = self.lock_state()?;
        let stale_request_ids = state
            .task_by_request
            .keys()
            .filter(|request_id| !live_request_ids.contains(request_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let mut released = false;
        for request_id in stale_request_ids {
            released |= Self::release_from_state(&mut state, &request_id)?;
        }
        Ok(released)
    }

    fn notify_state_changed(&self) {
        self.state_changed
            .send_modify(|generation| *generation = generation.wrapping_add(1));
    }
}

#[async_trait]
impl FlowControlPolicy for TaskWidthPolicy {
    async fn classify(
        &self,
        request: ClassifyRequest,
    ) -> Result<Classification, FlowControlPolicyError> {
        let (Some(request_id), Some(session_context)) =
            (request.request_id(), request.session_context())
        else {
            return Ok(Classification::default());
        };
        let request_id = request_id.to_owned();
        let task_id = session_context.session_id().to_owned();
        let mut state_changed = self.state_changed.subscribe();

        loop {
            {
                let mut state = self.lock_state()?;
                if let Some(existing_task_id) = state.task_by_request.get(&request_id) {
                    if existing_task_id == &task_id {
                        return Ok(Classification::default());
                    }
                    return Err(FlowControlPolicyError::new(format!(
                        "request {request_id} was already classified for another task"
                    )));
                }

                let active = state.active_by_task.entry(task_id.clone()).or_default();
                if *active < self.max_active_requests_per_task {
                    *active += 1;
                    state.task_by_request.insert(request_id, task_id);
                    return Ok(Classification::default());
                }
            }

            state_changed.changed().await.map_err(|_| {
                FlowControlPolicyError::new("task-width policy notification channel closed")
            })?;
        }
    }

    async fn on_event(&self, event: FlowControlEvent) -> Result<(), FlowControlPolicyError> {
        let released = match event {
            FlowControlEvent::Completed { request_id, .. }
            | FlowControlEvent::Aborted { request_id } => self.release(&request_id)?,
            FlowControlEvent::Reconcile { live_request_ids } => {
                self.reconcile(&live_request_ids)?
            }
            _ => false,
        };
        if released {
            self.notify_state_changed();
        }
        Ok(())
    }

    async fn teardown(&self) -> Result<(), FlowControlPolicyError> {
        let mut state = self.lock_state()?;
        state.active_by_task.clear();
        state.task_by_request.clear();
        drop(state);
        self.notify_state_changed();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use dynamo_kv_router::scheduling::SessionContext;

    use super::*;

    fn request(request_id: &str, task_id: &str) -> ClassifyRequest {
        ClassifyRequest::new(Instant::now(), 8, 0, 8)
            .with_request_id(request_id)
            .with_session_context(SessionContext::new(
                task_id.to_owned(),
                None,
                None,
                None,
                None,
            ))
    }

    #[tokio::test]
    async fn holds_same_task_until_completion() {
        let policy = Arc::new(TaskWidthPolicy::new(NonZeroUsize::MIN));
        policy
            .classify(request("request-1", "task-1"))
            .await
            .unwrap();

        let second = policy.classify(request("request-2", "task-1"));
        tokio::pin!(second);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut second)
                .await
                .is_err()
        );

        policy
            .on_event(FlowControlEvent::Completed {
                request_id: "request-1".to_owned(),
            })
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_millis(100), &mut second)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn completion_releases_only_one_of_multiple_waiters() {
        let policy = Arc::new(TaskWidthPolicy::new(NonZeroUsize::MIN));
        policy
            .classify(request("request-1", "task-1"))
            .await
            .unwrap();
        let second = policy.classify(request("request-2", "task-1"));
        let third = policy.classify(request("request-3", "task-1"));
        tokio::pin!(second, third);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut second)
                .await
                .is_err()
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut third)
                .await
                .is_err()
        );

        policy
            .on_event(FlowControlEvent::Completed {
                request_id: "request-1".to_owned(),
            })
            .await
            .unwrap();
        let second_won = tokio::select! {
            result = &mut second => {
                result.unwrap();
                true
            }
            result = &mut third => {
                result.unwrap();
                false
            }
        };

        let (winner, remaining) = if second_won {
            ("request-2", &mut third)
        } else {
            ("request-3", &mut second)
        };
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut *remaining)
                .await
                .is_err()
        );
        policy
            .on_event(FlowControlEvent::Completed {
                request_id: winner.to_owned(),
            })
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_millis(100), remaining)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn reconcile_cleans_state_for_requests_no_longer_owned_by_router() {
        let policy = Arc::new(TaskWidthPolicy::new(NonZeroUsize::MIN));
        policy
            .classify(request("request-1", "task-1"))
            .await
            .unwrap();
        let second = policy.classify(request("request-2", "task-1"));
        tokio::pin!(second);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut second)
                .await
                .is_err()
        );

        policy
            .on_event(FlowControlEvent::Reconcile {
                live_request_ids: Arc::from([]),
            })
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_millis(100), &mut second)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn allows_different_tasks_concurrently() {
        let policy = TaskWidthPolicy::new(NonZeroUsize::MIN);
        policy
            .classify(request("request-1", "task-1"))
            .await
            .unwrap();
        policy
            .classify(request("request-2", "task-2"))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn abort_releases_task_capacity() {
        let policy = TaskWidthPolicy::new(NonZeroUsize::MIN);
        policy
            .classify(request("request-1", "task-1"))
            .await
            .unwrap();
        policy
            .on_event(FlowControlEvent::Aborted {
                request_id: "request-1".to_owned(),
            })
            .await
            .unwrap();
        policy
            .classify(request("request-2", "task-1"))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn passes_through_requests_without_session_metadata() {
        let policy = TaskWidthPolicy::new(NonZeroUsize::MIN);
        let request = ClassifyRequest::new(Instant::now(), 8, 0, 8).with_request_id("request-1");
        assert_eq!(
            policy.classify(request).await.unwrap(),
            Classification::default()
        );
    }
}
