from pydantic import BaseModel, Field
from typing import List
from src.constant import HEIGHT, WIDTH, GUIDANCE_SCALE, NUM_INFERENCE_STEPS

class GenerateImageRequest(BaseModel):
    prompt: str = Field("Create a cat", description="The text prompt to guide the image generation process")
    height: int = Field(HEIGHT, description="The height of the generated image")
    width: int = Field(WIDTH, description="The width of the generated image")
    guidance_scale: float = Field(GUIDANCE_SCALE, description="The guidance scale for the image generation process")
    num_inference_steps: int = Field(NUM_INFERENCE_STEPS, description="The number of inference steps for the image generation process")


class BatchGenerateImageRequest(BaseModel):
    prompt: List[str] = Field(..., description="The text prompt to guide the image generation process")
    height: int = Field(HEIGHT, description="The height of the generated image")
    width: int = Field(WIDTH, description="The width of the generated image")
    guidance_scale: float = Field(GUIDANCE_SCALE, description="The guidance scale for the image generation process")
    num_inference_steps: int = Field(NUM_INFERENCE_STEPS, description="The number of inference steps for the image generation process")



