from huggingface_hub import snapshot_download
from src.constant import MODEL_ID, SAVE_MODEL_PATH

snapshot_download(repo_id=MODEL_ID, local_dir=SAVE_MODEL_PATH)
 