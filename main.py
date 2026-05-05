from fastapi import (
    FastAPI, 
    status, 
    HTTPException, 
    Depends, 
    Form, 
    UploadFile, 
    File,
    Request
)

import base64
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uuid

from src.schemas import GenerateImageRequest
from src.inference import Image2ImagePipeline
from fastapi.middleware.cors import CORSMiddleware
from src.utils.sercurity import get_api_key
from src.utils.logger import get_logger, setup_logging
from src.queue_img2img import Image2ImageQueue
from src.constant import (
    PORT,
    HEIGHT,
    WIDTH,
    GUIDANCE_SCALE,
    NUM_INFERENCE_STEPS
)

setup_logging()
logger = get_logger(__name__)

queue_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global queue_instance
    logger.info("Loading model pipeline...")
    pipeline = Image2ImagePipeline()
    queue_instance = Image2ImageQueue(pipeline)
    queue_instance.start()
    yield
    await queue_instance.stop()

app = FastAPI(
    title="Image2Image API",
    description="An API for generating images from prompts and optional input images using a pre-trained model",
    version="1.0.0",
    root_path="/api/v1",
    dependencies=[Depends(get_api_key)],
    lifespan=lifespan
)

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"

app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Web interface",
    description="Serve HTML interface",
)
async def root():
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="A simple health check endpoint",
)
async def health_check():
    return {"message": "Image2Image API is running"}


@app.post(
    path="/generate", 
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate image",
    description="Generate a new image based on a text prompt"
)
async def generate_image(request: GenerateImageRequest):
    task_id = str(uuid.uuid4())
    await queue_instance.set_task_state(task_id, "pending")

    task_payload = {
        "task_id": task_id,
        "operation": "generate",
        "kwargs": {
            "prompt": request.prompt,
            "height": request.height,
            "width": request.width,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps
        }
    }
    await queue_instance.put_task(task_payload)
    
    return {"task_id": task_id, "message": "Task queued"}


@app.post(
    path="/edit", 
    status_code=status.HTTP_202_ACCEPTED,
    summary="Edit image",
    description="Edit an image based on a text prompt and an input image"    
)
async def edit_image(
    prompt: str = Form(default="An image of a cat wearing a hat", description="The text prompt to guide the image editing process"),
    image: UploadFile = File(..., description="The image file to be edited"),
    height: int = Form(HEIGHT, description="The height of the generated image"),
    width: int = Form(WIDTH, description="The width of the generated image"),
):
    if not image:
        raise HTTPException(status_code=400, detail="Image is required for edit")

    image_bytes = await image.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    task_id = str(uuid.uuid4())
    await queue_instance.set_task_state(task_id, "pending")

    task_payload = {
        "task_id": task_id,
        "operation": "edit",
        "kwargs": {
            "prompt": prompt,
            "image": image_b64,
            "height": height,
            "width": width,
            "guidance_scale": GUIDANCE_SCALE,
            "num_inference_steps": NUM_INFERENCE_STEPS,
        },
    }
    await queue_instance.put_task(task_payload)

    return {"task_id": task_id, "message": "Task queued"}


@app.get(
    path="/result/{task_id}", 
    summary="Get task result",
    description="Get the result of a specific task by its ID"
)
async def get_result(task_id: str):
    result = await queue_instance.get_task_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    return result


# get pending tasks count for monitoring
@app.get(
    path="/tasks/pending", 
    summary="Get pending tasks count",
    description="Get the count of pending tasks in the queue"
)
async def get_pending_tasks():
    count = queue_instance.get_pending_tasks_count()
    return {"pending_tasks": count}


@app.get(
    path="/tasks/stats",
    summary="Get task stats by status",
    description="Get queue/task statistics grouped by status: pending, processing, completed, failed"
)
async def get_task_stats():
    return await queue_instance.get_task_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
