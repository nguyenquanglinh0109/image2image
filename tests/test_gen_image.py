import requests
import threading
import time
import random
import traceback
import base64
from pathlib import Path
import datetime
from dotenv import load_dotenv
import os

load_dotenv()


BASE_URL = "https://n4q401fwx0j8ya-8060.proxy.runpod.net/api/v1"
GENERATE_URL = f"{BASE_URL}/generate"
RESULT_URL = f"{BASE_URL}/result"
OUTPUT_DIR = Path("test_image")
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = os.getenv("API_KEY_NAME")
"""
    
        A fantasy landscape with mountains and a river, A futuristic cityscape at sunset, A serene beach with palm trees and clear blue water, A bustling market in a medieval town, A majestic castle on a hilltop surrounded by clouds, A cozy cabin in a snowy forest, A vibrant coral reef teeming with colorful fish,A surreal dreamscape with floating islands and waterfalls

        A whimsical scene of animals engaging in human activities,A surreal landscape that combines elements of nature and technology,a forest with trees made of circuit boards or a beach with waves made of binary code,Anime style portrait of a young person

    """
    

def build_payload(prompt=None):
    if not prompt:
        prompt = random.choice([
            "A fantasy landscape with mountains and a river",
        ])

    payload = {
        "prompt": prompt,
        "height": 512,
        "width": 512,
        "num_inference_steps": 4,
    }

    return payload


lock = threading.Lock()
success = 0
fail = 0


def build_headers():
    if not API_KEY_NAME or not API_KEY:
        raise RuntimeError("Missing API credentials. Please set API_KEY_NAME and API_KEY in .env")

    return {
        "Content-Type": "application/json",
        API_KEY_NAME: API_KEY,
    }


def wait_for_result(task_id: str, headers: dict, timeout: int = 180, interval: float = 0.5):
    deadline = time.time() + timeout

    while time.time() < deadline:
        res = requests.get(f"{RESULT_URL}/{task_id}", timeout=30, headers=headers)
        if res.status_code != 200:
            raise RuntimeError(f"Poll failed for {task_id}: status={res.status_code}, body={res.text}")

        data = res.json()
        status = data.get("status")

        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"Task {task_id} failed: {data.get('error')}")

        time.sleep(interval)

    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")

def send_request():
    global success, fail

    payload = build_payload()
    headers = build_headers()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        res = requests.post(
            GENERATE_URL, 
            json=payload, 
            timeout=60,
            headers=headers
        )
        with open(f"images/a_{int(time.time() * 1000)}.text", "w") as f:
            f.write(str(res.json()))

        # New API flow: queue first, then poll by task_id
        if res.status_code == 202:
            task_id = res.json().get("task_id")
            if not task_id:
                raise RuntimeError(f"Missing task_id in response: {res.text}")

            result = wait_for_result(task_id, headers=headers)
            base64_image = result.get("image")
            if not base64_image:
                raise RuntimeError(f"Task {task_id} completed but image is missing")

            output_path = OUTPUT_DIR / f"batch_{task_id}_{int(time.time() * 1000)}.png"
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(base64_image))

            with lock:
                success += 1
            print(f"[OK] task_id={task_id} saved={output_path}")
            return

        # Backward-compatible flow: old API returns image directly
        with lock:
            if res.status_code == 201:
                success += 1
                base64_image = res.json().get("image")
                print(f"Received image of size: {len(base64_image)} bytes, time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                if base64_image:
                    filename = datetime.now().strftime("a_%H-%M-%S.png")
                    with open(OUTPUT_DIR / filename, "wb") as f:
                        f.write(base64.b64decode(base64_image))

            else:
                fail += 1
                print(f"Status code: {res.status_code}")

    except Exception as e:
        with lock:
            fail += 1
            print(f"Exception: {e}")
            traceback.print_exc()


def sample_request():
    payload = build_payload()
    headers = build_headers()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        res = requests.post(GENERATE_URL, json=payload, timeout=60, headers=headers)
        with open("a.text", "w") as f:
            f.write(str(res.json()))

        if res.status_code == 202:
            task_id = res.json().get("task_id")
            if not task_id:
                raise RuntimeError(f"Missing task_id in response: {res.text}")

            result = wait_for_result(task_id, headers=headers)
            base64_image = result.get("image")
            if not base64_image:
                raise RuntimeError(f"Task {task_id} completed but image is missing")

            output_path = OUTPUT_DIR / f"sample_{task_id}_{int(time.time() * 1000)}.png"
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(base64_image))

            print(f"[SAMPLE] task_id={task_id} saved={output_path}")
            return

        if res.status_code == 201:
            base64_image = res.json().get("image")
            print(f"Received image of size: {len(base64_image)} bytes, current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            if base64_image:
                filename = datetime.now().strftime("sample_%H-%M-%S.png")
                with open(OUTPUT_DIR / filename, "wb") as f:
                    f.write(base64.b64decode(base64_image))
        else:
            print(f"Status code: {res.status_code}")
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    ## ====== Stress test with 20 concurrent requests =======
    num_threads = 10
    threads = []

    start_time = time.time()

    for _ in range(num_threads):
        thread = threading.Thread(target=send_request)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    end_time = time.time()
    duration = end_time - start_time

    print(f"Total requests: {num_threads}, Success: {success}, Fail: {fail}, Duration: {duration:.2f} seconds")


    ## ======= Sample request =======
    # sample_request()