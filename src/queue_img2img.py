import asyncio
import time
import base64
from collections import defaultdict
from src.utils.logger import get_logger
from src.utils.convert_to_pil import convert_bytes_to_pil
from src.constant import (
    QUEUE_BATCH_MAX_SIZE, 
    QUEUE_BATCH_MAX_WAIT_MS, 
    TASK_RESULT_TTL_SECONDS,
    TASK_RESULT_CLEANUP_INTERVAL_SECONDS,
    MAX_TASK_RESULTS
)

logger = get_logger(__name__)

class Image2ImageQueue:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.fifo_queue = asyncio.Queue()
        self.task_results = {}
        self.task_results_lock = asyncio.Lock()
        self.worker_task = None
        self.cleanup_task = None

    def _now_ts(self) -> float:
        return time.time()

    async def _enforce_task_results_limit_locked(self):
        if MAX_TASK_RESULTS <= 0:
            return

        removed_count = 0
        while len(self.task_results) > MAX_TASK_RESULTS:
            terminal_candidates = [
                (task_id, data)
                for task_id, data in self.task_results.items()
                if data.get("status") in {"completed", "failed"}
            ]

            candidates = terminal_candidates or list(self.task_results.items())
            oldest_task_id, _ = min(
                candidates,
                key=lambda item: (
                    item[1].get("done_at") or item[1].get("created_at") or float("inf")
                ),
            )
            self.task_results.pop(oldest_task_id, None)
            removed_count += 1

        if removed_count:
            logger.info(
                "Evicted %s old task result(s) due to MAX_TASK_RESULTS=%s",
                removed_count,
                MAX_TASK_RESULTS,
            )

    async def set_task_state(self, task_id: str, status_value: str, **extra_fields):
        now = self._now_ts()
        async with self.task_results_lock:
            existing = self.task_results.get(task_id, {})
            created_at = existing.get("created_at", now)
            result_data = {
                "status": status_value,
                "created_at": created_at,
                "updated_at": now,
                "done_at": existing.get("done_at"),
            }
            result_data.update(extra_fields)

            if status_value in {"completed", "failed"}:
                result_data["done_at"] = now

            self.task_results[task_id] = result_data
            await self._enforce_task_results_limit_locked()

    async def get_task_result(self, task_id: str):
        async with self.task_results_lock:
            result = self.task_results.get(task_id)
            if not result:
                return None

            # Return a copy so callers cannot mutate in-memory queue state.
            response = dict(result)
            response["task_id"] = task_id
            return response

    async def get_task_stats(self) -> dict:
        counts = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
        }

        async with self.task_results_lock:
            for data in self.task_results.values():
                status = data.get("status")
                if status in counts:
                    counts[status] += 1

        counts["waiting_in_queue"] = self.fifo_queue.qsize()
        counts["total_tracked"] = sum(counts[key] for key in ("pending", "processing", "completed", "failed"))
        return counts

    async def put_task(self, task_payload: dict):
        await self.fifo_queue.put(task_payload)

    # def get_pending_tasks_count(self) -> int:
    #     return self.fifo_queue.qsize()

    async def _cleanup_task_results(self):
        logger.info(
            "Task results cleanup started (ttl=%ss, interval=%ss, max=%s)",
            TASK_RESULT_TTL_SECONDS,
            TASK_RESULT_CLEANUP_INTERVAL_SECONDS,
            MAX_TASK_RESULTS,
        )

        while True:
            await asyncio.sleep(TASK_RESULT_CLEANUP_INTERVAL_SECONDS)
            now = self._now_ts()
            cutoff = now - TASK_RESULT_TTL_SECONDS

            async with self.task_results_lock:
                expired_task_ids = [
                    task_id
                    for task_id, data in self.task_results.items()
                    if data.get("status") in {"completed", "failed"}
                    and data.get("done_at") is not None
                    and data["done_at"] < cutoff
                ]

                for task_id in expired_task_ids:
                    self.task_results.pop(task_id, None)

                if expired_task_ids:
                    logger.info(
                        "Cleaned up %s expired task result(s)",
                        len(expired_task_ids),
                    )

                await self._enforce_task_results_limit_locked()

    def _task_signature(self, task_data: dict) -> tuple:
        kwargs = task_data["kwargs"]
        return (
            task_data["operation"],
            kwargs["height"],
            kwargs["width"],
            kwargs["guidance_scale"],
            kwargs["num_inference_steps"],
        )

    async def _process_task_group(self, tasks: list[dict]):
        operation = tasks[0]["operation"]
        first_kwargs = tasks[0]["kwargs"]
        prompts = [task["kwargs"]["prompt"] for task in tasks]

        try:
            logger.info(f"Processing grouped batch op={operation} size={len(tasks)}")
            if operation == "generate":
                images = await self.pipeline.generate_batch_image(
                    prompt=prompts,
                    height=first_kwargs["height"],
                    width=first_kwargs["width"],
                    guidance_scale=first_kwargs["guidance_scale"],
                    num_inference_steps=first_kwargs["num_inference_steps"],
                )
            elif operation == "edit":
                pil_images = []
                for task in tasks:
                    image_data = base64.b64decode(task["kwargs"]["image"])
                    pil_image = await convert_bytes_to_pil(image_data)
                    pil_images.append(pil_image)

                images = await self.pipeline.edit_batch_image(
                    prompt=prompts,
                    image=pil_images,
                    height=first_kwargs["height"],
                    width=first_kwargs["width"],
                    guidance_scale=first_kwargs["guidance_scale"],
                    num_inference_steps=first_kwargs["num_inference_steps"],
                )
            else:
                logger.error(f"Unsupported operation in task group: {operation}")
                raise RuntimeError(f"Unsupported operation: {operation}")

            if len(images) != len(tasks):
                logger.error(
                    f"Batch output mismatch: expected {len(tasks)} images, got {len(images)}"
                )
                raise RuntimeError(
                    f"Batch output mismatch: expected {len(tasks)} images, got {len(images)}"
                )

            for task, image in zip(tasks, images):
                await self.set_task_state(task["task_id"], "completed", image=image)

        except Exception as batch_error:
            logger.error(f"Grouped batch failed (op={operation}, size={len(tasks)}): {batch_error}")

            for task in tasks:
                task_id = task["task_id"]
                kwargs = task["kwargs"]
                try:
                    if operation == "generate":
                        image = await self.pipeline.generate_single_image(
                            prompt=kwargs["prompt"],
                            height=kwargs["height"],
                            width=kwargs["width"],
                            guidance_scale=kwargs["guidance_scale"],
                            num_inference_steps=kwargs["num_inference_steps"],
                        )
                    elif operation == "edit":
                        image_data = base64.b64decode(kwargs["image"])
                        pil_image = await convert_bytes_to_pil(image_data)
                        image = await self.pipeline.edit_single_image(
                            prompt=kwargs["prompt"],
                            image=pil_image,
                            height=kwargs["height"],
                            width=kwargs["width"],
                            guidance_scale=kwargs["guidance_scale"],
                            num_inference_steps=kwargs["num_inference_steps"],
                        )
                    else:
                        raise RuntimeError(f"Unsupported operation: {operation}")

                    await self.set_task_state(task_id, "completed", image=image)
                except Exception as single_error:
                    logger.error(f"Task {task_id} failed: {single_error}")
                    await self.set_task_state(task_id, "failed", error=str(single_error))


    async def _fifo_worker(self):
        logger.info("Starting FIFO worker...")
        max_batch_size = QUEUE_BATCH_MAX_SIZE
        max_wait_ms = QUEUE_BATCH_MAX_WAIT_MS

        while True:
            batch = []
            first_task = await self.fifo_queue.get()
            batch.append(first_task)

            deadline = asyncio.get_running_loop().time() + (max_wait_ms / 1000)
            while len(batch) < max_batch_size:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break

                try:
                    next_task = await asyncio.wait_for(self.fifo_queue.get(), timeout=timeout)
                    batch.append(next_task)
                except asyncio.TimeoutError:
                    break

            for task in batch:
                await self.set_task_state(task["task_id"], "processing")

            grouped_tasks = defaultdict(list)
            for task in batch:
                grouped_tasks[self._task_signature(task)].append(task)

            for _, tasks in grouped_tasks.items():
                await self._process_task_group(tasks)

            for _ in batch:
                self.fifo_queue.task_done()

    def start(self):
        self.worker_task = asyncio.create_task(self._fifo_worker())
        self.cleanup_task = asyncio.create_task(self._cleanup_task_results())

    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        from contextlib import suppress
        with suppress(asyncio.CancelledError):
            if self.worker_task:
                await self.worker_task
            if self.cleanup_task:
                await self.cleanup_task
