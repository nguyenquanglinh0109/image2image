import base64
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constant import API_KEY, API_KEY_NAME

BASE_URL = "https://n4q401fwx0j8ya-8060.proxy.runpod.net"
GENERATE_URL = f"{BASE_URL}/generate"
RESULT_URL = f"{BASE_URL}/result"
OUTPUT_DIR = Path("test_image")

NUM_REQUESTS = 10
POLL_INTERVAL_SECONDS = 0.5
POLL_TIMEOUT_SECONDS = 180

PROMPTS = [
    "A fantasy landscape with mountains and a river",
    "A futuristic cityscape at sunset",
    "A serene beach with palm trees and clear blue water",
    "A bustling market in a medieval town",
    "A majestic castle on a hilltop surrounded by clouds",
    "A cozy cabin in a snowy forest",
    "A vibrant coral reef teeming with colorful fish",
    "A surreal dreamscape with floating islands and waterfalls",
    "A peaceful countryside with rolling hills and a winding road",
    "A magical forest with glowing mushrooms and fireflies",
    "Anime style portrait of a young person",
]

counter_lock = threading.Lock()
success_count = 0
fail_count = 0


def build_payload(prompt: str | None = None) -> dict:
    final_prompt = prompt or random.choice(PROMPTS)
    return {
        "prompt": final_prompt,
        "height": 512,
        "width": 512,
        "num_inference_steps": 4,
    }


def submit_generate_task(session: requests.Session, payload: dict) -> str:
    headers = {API_KEY_NAME: API_KEY}
    res = session.post(GENERATE_URL, json=payload, headers=headers, timeout=60)

    if res.status_code != 202:
        raise RuntimeError(
            f"Expected 202 from /generate, got {res.status_code}. body={res.text}"
        )

    data = res.json()
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Missing task_id in response: {data}")
    return task_id


def wait_for_result(
    session: requests.Session,
    task_id: str,
    timeout_seconds: int = POLL_TIMEOUT_SECONDS,
    interval_seconds: float = POLL_INTERVAL_SECONDS,
) -> dict:
    deadline = time.time() + timeout_seconds
    headers = {API_KEY_NAME: API_KEY}

    while time.time() < deadline:
        res = session.get(f"{RESULT_URL}/{task_id}", headers=headers, timeout=30)
        if res.status_code != 200:
            raise RuntimeError(
                f"Poll failed for task_id={task_id}: status={res.status_code}, body={res.text}"
            )

        data = res.json()
        status = data.get("status")

        if status == "completed":
            return data

        if status == "failed":
            raise RuntimeError(
                f"Task failed task_id={task_id}, error={data.get('error')}"
            )

        time.sleep(interval_seconds)

    raise TimeoutError(f"Task timed out task_id={task_id} after {timeout_seconds}s")


def save_result_image(task_id: str, result: dict) -> Path:
    image_b64 = result.get("image")
    if not image_b64:
        raise RuntimeError(f"Completed task has no image: task_id={task_id}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"stress_{task_id}_{int(time.time() * 1000)}.png"
    output_path.write_bytes(base64.b64decode(image_b64))
    return output_path


def run_single_request(request_index: int) -> tuple[bool, str]:
    session = requests.Session()
    payload = build_payload()

    try:
        task_id = submit_generate_task(session, payload)
        result = wait_for_result(session, task_id)
        image_path = save_result_image(task_id, result)
        return True, f"[OK] idx={request_index} task_id={task_id} saved={image_path}"
    except Exception as exc:
        traceback.print_exc()
        return False, f"[FAIL] idx={request_index} error={exc}"
    finally:
        session.close()


def main():
    global success_count, fail_count

    start_time = time.time()
    print(f"Starting stress test with {NUM_REQUESTS} concurrent requests")

    with ThreadPoolExecutor(max_workers=NUM_REQUESTS) as executor:
        futures = [executor.submit(run_single_request, i + 1) for i in range(NUM_REQUESTS)]

        for future in as_completed(futures):
            ok, message = future.result()
            with counter_lock:
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
            print(message)

    duration = time.time() - start_time
    print(
        f"Total requests={NUM_REQUESTS}, success={success_count}, "
        f"fail={fail_count}, duration={duration:.2f}s"
    )


if __name__ == "__main__":
    main()