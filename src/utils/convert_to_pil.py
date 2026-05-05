from PIL import Image
from io import BytesIO
from fastapi import HTTPException

async def convert_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Async image processing"""
    try:
        
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")