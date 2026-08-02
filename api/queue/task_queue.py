"""
api/queue/task_queue.py — Async Task Queue
Step 1.3: Concurrent Session Controller + Async Task Queue (Local FastAPI Orchestration Server)

Handles 25-30 concurrent users by:
- Queuing requests when LLM is busy
- Processing in parallel where possible
- Tracking progress for long-running queries
- Preventing UI freeze via async processing
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A single queued task."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = ""
    user_id: str = "anonymous"
    question: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    wait_time: float = 0.0
    exec_time: float = 0.0

    @property
    def total_time(self) -> float:
        return self.wait_time + self.exec_time

    def to_dict(self) -> dict:
        return {
            "task_id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "question": self.question,
            "status": self.status.value,
            "wait_time": round(self.wait_time, 2),
            "exec_time": round(self.exec_time, 2),
            "total_time": round(self.total_time, 2),
            "created_at": self.created_at.isoformat(),
        }


class TaskQueue:
    """
    Async task queue for handling concurrent AI requests.

    Architecture:
    User Request → Queue → Worker Pool → LLM → Response

    With 25-30 users:
    - max_workers = 3-5 (LLM inference slots)
    - Others wait in queue with position tracking
    - Each user gets async response (no blocking)
    """

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self._queue: asyncio.Queue = asyncio.Queue()
        self._tasks: dict[str, Task] = {}
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._stats = {
            "total_processed": 0,
            "total_failed": 0,
            "avg_wait_time": 0.0,
            "avg_exec_time": 0.0,
        }

    async def start(self):
        """Start the worker pool."""
        self._running = True
        self._semaphore = asyncio.Semaphore(self.max_workers)

        # Start worker coroutines
        for i in range(self.max_workers):
            worker = asyncio.create_task(
                self._worker(f"worker-{i+1}")
            )
            self._workers.append(worker)

        logger.info(
            f"Task queue started with {self.max_workers} workers"
        )

    async def stop(self):
        """Stop the worker pool gracefully."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()
        logger.info("Task queue stopped")

    async def submit(
        self,
        question: str,
        fn: Callable,
        session_id: str = "",
        user_id: str = "anonymous",
        **kwargs,
    ) -> Task:
        """
        Submit a query task to the queue.
        Returns Task immediately — caller polls for result.
        """
        task = Task(
            session_id=session_id or str(uuid.uuid4())[:8],
            user_id=user_id,
            question=question,
        )
        self._tasks[task.id] = task

        # Put in queue
        await self._queue.put((task, fn, kwargs))

        queue_position = self._queue.qsize()
        logger.info(
            f"Task {task.id} queued: '{question[:40]}' "
            f"(position: {queue_position})"
        )

        return task

    async def submit_and_wait(
        self,
        question: str,
        fn: Callable,
        session_id: str = "",
        user_id: str = "anonymous",
        timeout: float = 600.0,
        **kwargs,
    ) -> Task:
        """
        Submit task and wait for completion.
        Used for simple request-response pattern.
        """
        task = await self.submit(
            question=question,
            fn=fn,
            session_id=session_id,
            user_id=user_id,
            **kwargs,
        )

        # Poll for completion
        start = time.time()
        while time.time() - start < timeout:
            if task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ):
                return task
            await asyncio.sleep(0.5)

        # Timeout
        task.status = TaskStatus.FAILED
        task.error = f"Task timed out after {timeout}s"
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID for polling."""
        return self._tasks.get(task_id)

    def get_queue_stats(self) -> dict:
        """Get current queue statistics."""
        pending = sum(
            1 for t in self._tasks.values()
            if t.status == TaskStatus.PENDING
        )
        running = sum(
            1 for t in self._tasks.values()
            if t.status == TaskStatus.RUNNING
        )
        return {
            "queue_size": self._queue.qsize(),
            "pending_tasks": pending,
            "running_tasks": running,
            "max_workers": self.max_workers,
            "total_processed": self._stats["total_processed"],
            "total_failed": self._stats["total_failed"],
            "avg_wait_time_s": round(self._stats["avg_wait_time"], 2),
            "avg_exec_time_s": round(self._stats["avg_exec_time"], 2),
        }

    async def _worker(self, worker_name: str):
        """Worker coroutine — processes tasks from queue."""
        logger.debug(f"{worker_name} started")

        while self._running:
            try:
                # Get next task (with timeout to allow shutdown)
                try:
                    task, fn, kwargs = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Calculate wait time
                task.wait_time = (
                    datetime.now() - task.created_at
                ).total_seconds()
                task.started_at = datetime.now()
                task.status = TaskStatus.RUNNING

                logger.info(
                    f"{worker_name} processing task {task.id}: "
                    f"'{task.question[:40]}' "
                    f"(waited {task.wait_time:.1f}s)"
                )

                # Execute task
                exec_start = time.time()
                try:
                    # Run in thread pool to not block event loop
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: fn(
                            question=task.question,
                            user_id=task.user_id,
                            session_id=task.session_id,
                            **kwargs,
                        )
                    )
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    self._stats["total_processed"] += 1

                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    self._stats["total_failed"] += 1
                    logger.error(
                        f"Task {task.id} failed: {e}", exc_info=True
                    )

                finally:
                    task.exec_time = time.time() - exec_start
                    task.completed_at = datetime.now()
                    self._queue.task_done()

                    # Update stats
                    n = self._stats["total_processed"]
                    if n > 0:
                        self._stats["avg_wait_time"] = (
                            (self._stats["avg_wait_time"] * (n-1) +
                             task.wait_time) / n
                        )
                        self._stats["avg_exec_time"] = (
                            (self._stats["avg_exec_time"] * (n-1) +
                             task.exec_time) / n
                        )

                    logger.info(
                        f"Task {task.id} {task.status.value}: "
                        f"wait={task.wait_time:.1f}s "
                        f"exec={task.exec_time:.1f}s"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{worker_name} error: {e}")

        logger.debug(f"{worker_name} stopped")


# ─── Singleton ────────────────────────────────────────────────────────────────
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    global _task_queue
    if _task_queue is None:
        from erp42_analytics.core.config import config
        # Scale workers based on available CPU/GPU
        workers = getattr(config.app, "max_workers", 3)
        _task_queue = TaskQueue(max_workers=workers)
    return _task_queue
