import asyncio
from time import time

import torch
import diffusers
from sdnq import SDNQConfig # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from PIL import Image
from typing import List, Optional
import base64
from io import BytesIO
import os
import gc
from huggingface_hub import snapshot_download

from src.constant import (
    NUM_INFERENCE_STEPS, 
    GUIDANCE_SCALE, 
    HEIGHT, 
    WIDTH,
    MODEL_ID,
    SAVE_MODEL_PATH,
    MEMORY_CLEANUP_INTERVAL
)
from src.utils.logger import get_logger
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = get_logger(__name__)

class Image2ImagePipeline:
    def __init__(self):
        self.device = self._get_device()
        self.request_counter = 0
        self.memory_cleanup_interval = MEMORY_CLEANUP_INTERVAL
        self.pipe = self._load_model()
        # Warmup is optional and can be done manually if needed
        self._warmup()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"

    def _cleanup_memory(self, force: bool = False):
        self.request_counter += 1
        should_cleanup = force or (self.request_counter % self.memory_cleanup_interval == 0)
        if not should_cleanup:
            return

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif self.device == "xpu":
            torch.xpu.empty_cache()

    def _run_pipe(self, **kwargs):
        """Run a blocking model forward pass synchronously.

        This method is executed in a worker thread via asyncio.to_thread
        from async request handlers.
        """
        with torch.inference_mode():
            return self.pipe(**kwargs)

    def _load_model(self):
        """Load the pre-trained model pipeline."""    
        try:

            if not SAVE_MODEL_PATH:
                logger.error("SAVE_MODEL_PATH is not set. Please set it to a valid directory path.")
                raise ValueError("SAVE_MODEL_PATH is not set.")

            snapshot_download(repo_id=MODEL_ID, local_dir=SAVE_MODEL_PATH)

            pipe = diffusers.Flux2KleinPipeline.from_pretrained(
                SAVE_MODEL_PATH, 
                torch_dtype=torch.bfloat16
            ).to(self.device)

            # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
            if triton_is_available and self.device in {"cuda", "xpu"}:
                pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
                pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
                enable_torch_compile = os.getenv("ENABLE_TORCH_COMPILE", "0") == "1"
                if enable_torch_compile:
                    pipe.transformer = torch.compile(pipe.transformer) # optional for faster speeds
                else:
                    logger.info("Torch compile disabled to avoid long-run VRAM growth from graph cache.")

            # pipe.enable_model_cpu_offload()
            return pipe

        except Exception as e:
            
            logger.error(f"Failed to load the model pipeline: {e}")
            raise RuntimeError(f"Failed to load the model pipeline: {e}")


    def _warmup(self):
        """Warm up the model pipeline by running a dummy inference."""
        try:
            logger.info("Warming up the model pipeline...")
            with torch.inference_mode():
                self.pipe(
                    prompt="A warm-up image",
                    height=HEIGHT,
                    width=WIDTH,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    generator=torch.manual_seed(0)
                )
            self._cleanup_memory(force=True)
            logger.info("Model pipeline warmed up successfully.")
        
        except Exception as e:
            logger.error(f"Failed to warm up the model pipeline: {e}")
            raise RuntimeError(f"Failed to warm up the model pipeline: {e}")
      
    async def generate_single_image(
        self,
        prompt: str, 
        height: int = HEIGHT, 
        width: int = WIDTH, 
        guidance_scale: float = GUIDANCE_SCALE, 
        num_inference_steps: int = NUM_INFERENCE_STEPS, 
    ) -> str:
        """Generate image

        Args:
            prompt (str): The text prompt to guide the image generation process.
            height (int, optional): The height of the generated image. Defaults to HEIGHT.
            width (int, optional): The width of the generated image. Defaults to WIDTH.
            guidance_scale (float, optional): The guidance scale for the image generation process. Defaults to GUIDANCE_SCALE.
            num_inference_steps (int, optional): The number of inference steps for the image generation process. Defaults to NUM_INFERENCE_STEPS.

        Returns:
            str: The base64-encoded string of the generated image.
        """
        try:
            logger.info(f"Generating image with prompt: '{prompt}'")
            output = await asyncio.to_thread(
                self._run_pipe,
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                # generator=torch.manual_seed(0)
            )
            image = output.images[0]

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode()

            del image
            del buffer
            del output
            self._cleanup_memory()

            return encoded_image
            
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            raise RuntimeError(f"Failed to generate image: {e}")
  
    async def edit_single_image(
        self,
        prompt: str, 
        image: Image.Image, 
        height: int = HEIGHT, 
        width: int = WIDTH, 
        guidance_scale: float = GUIDANCE_SCALE, 
        num_inference_steps: int = NUM_INFERENCE_STEPS, 
    ) -> str:
        """Edit an image based on a text prompt.

        Args:
            prompt (str): The text prompt to guide the image editing process.
            image (Image.Image): The input image to be edited.
            height (int, optional): The height of the edited image. Defaults to HEIGHT.
            width (int, optional): The width of the edited image. Defaults to WIDTH.
            guidance_scale (float, optional): The guidance scale for the image editing process. Defaults to GUIDANCE_SCALE.
            num_inference_steps (int, optional): The number of inference steps for the image editing process. Defaults to NUM_INFERENCE_STEPS.

        Returns:
            str: The base64-encoded string of the edited image.
        """
        try:
            logger.info(f"Editing image with prompt: '{prompt}'")
            output = await asyncio.to_thread(
                self._run_pipe,
                prompt=prompt,
                image=image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            edited_image = output.images[0]

            buffer = BytesIO()
            edited_image.save(buffer, format="PNG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode()

            del edited_image
            del buffer
            del output
            self._cleanup_memory()

            return encoded_image
        
        except Exception as e:
            logger.error(f"Failed to edit image: {e}")
            raise RuntimeError(f"Failed to edit image: {e}")
        

    async def generate_batch_image(
        self,
        prompt: List[str], 
        height: int = HEIGHT, # Chuyển về int
        width: int = WIDTH,   # Chuyển về int
        guidance_scale: float = GUIDANCE_SCALE, 
        num_inference_steps: int = NUM_INFERENCE_STEPS, 
    ) -> List[str]: # Trả về List[str]
        try:
            logger.info(f"Generating batch of {len(prompt)} images")
            logger.info(f"Generate batch image with prompts: {prompt}")
            encoded_images = []

            outputs = await asyncio.to_thread(
                self._run_pipe,
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            images = outputs.images

            for image in images:
                with BytesIO() as buffer: # Dùng 'with' để tự động đóng buffer
                    image.save(buffer, format="PNG")
                    encoded_images.append(base64.b64encode(buffer.getvalue()).decode())

            # Dọn dẹp
            del images
            del outputs
            self._cleanup_memory()

            return encoded_images
            
        except Exception as e:
            logger.error(f"Failed to generate batch: {e}")
            raise RuntimeError(f"Failed to generate batch: {e}")
    


    async def edit_batch_image(
        self,
        prompt: List[str], 
        image: List[Image.Image], 
        height: int = HEIGHT, 
        width: int = WIDTH,   
        guidance_scale: float = GUIDANCE_SCALE, 
        num_inference_steps: int = NUM_INFERENCE_STEPS, 
    ) -> List[str]:
        """Edit batch image based on a text prompt.

        Args:
            prompt (str): The text prompt to guide the image editing process.
            image (Image.Image): The input image to be edited.
            height (int, optional): The height of the edited image. Defaults to HEIGHT.
            width (int, optional): The width of the edited image. Defaults to WIDTH.
            guidance_scale (float, optional): The guidance scale for the image editing process. Defaults to GUIDANCE_SCALE.
            num_inference_steps (int, optional): The number of inference steps for the image editing process. Defaults to NUM_INFERENCE_STEPS.

        Returns:
            str: The base64-encoded string of the edited image.
        """
        try:
            logger.info(f"Editing batch of {len(prompt)} images")
            logger.info(f"Editing batch images with prompts: {prompt}")
            encoded_images = []

            outputs = await asyncio.to_thread(
                self._run_pipe,
                prompt=prompt,
                image=image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            edited_image = outputs.images

            for img in edited_image:
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                encoded_image = base64.b64encode(buffer.getvalue()).decode()
                encoded_images.append(encoded_image)

            del edited_image
            del buffer
            del outputs
            self._cleanup_memory()

            return encoded_images
        
        except Exception as e:
            logger.error(f"Failed to edit image: {e}")
            raise RuntimeError(f"Failed to edit image: {e}")
        
if __name__ == "__main__":
    # ======= Single Image Generation Test =======
    # pipeline = Image2ImagePipeline()
    # prompt = "A beautiful landscape with mountains and a river"
    # image = Image.open("test_image/sample.png")
    # result = pipeline.edit_single_image(prompt=prompt, image=image)
    # print(result)


    # ======= Batch Image Generation Test =======
    pipeline = Image2ImagePipeline()
    prompts = [
        "A beautiful landscape with mountains and a river",
        "A futuristic cityscape at sunset",
    ]
    height = 512
    width = 1024 
    result = asyncio.run(pipeline.generate_batch_image(prompt=prompts, height=height, width=width))

    for i, encoded_image in enumerate(result):
        with open(f"test_image/edited_batch_{i}_{int(time.time() * 1000)}.png", "wb") as f:
            f.write(base64.b64decode(encoded_image))
