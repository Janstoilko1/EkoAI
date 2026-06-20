from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_TOKEN")
snapshot_download(
    repo_id="jakobrajsp/thrash_dataset",
    repo_type="dataset",        # ← pomembno, ker je dataset ne model
    token=token,
    local_dir="./dataset"       # kam se shrani
)