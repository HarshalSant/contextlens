"""
Push ContextLens to Hugging Face Spaces.

Run AFTER authenticating:
    hf auth login          # paste a write-scoped token from hf.co/settings/tokens

Then:
    python scripts/push_to_hf_space.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, whoami

REPO_ROOT = Path(__file__).parent.parent
SPACE_NAME = "contextlens"


def main() -> None:
    # Verify we're logged in
    try:
        user = whoami()
        hf_username = user["name"]
        print(f"Logged in as: {hf_username}")
    except Exception:
        print("ERROR: Not logged in. Run: hf auth login")
        raise SystemExit(1)

    api = HfApi()
    repo_id = f"{hf_username}/{SPACE_NAME}"

    # Create the Space (idempotent — safe to re-run)
    print(f"Creating/updating Space: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )

    # Files to upload to the Space
    files_to_upload = {
        # The HF README (with YAML frontmatter) goes as README.md
        REPO_ROOT / "hf_README.md": "README.md",
        REPO_ROOT / "app.py": "app.py",
        REPO_ROOT / "requirements.txt": "requirements.txt",
        # Package source
        REPO_ROOT / "src": "src",
        # Examples (demo logic imported by app.py)
        REPO_ROOT / "examples" / "demo.py": "examples/demo.py",
    }

    print("Uploading files...")
    for local_path, remote_path in files_to_upload.items():
        local = Path(local_path)
        if local.is_dir():
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="space",
            )
            print(f"  [dir]  {local.name}/ -> {remote_path}/")
        elif local.is_file():
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="space",
            )
            print(f"  [file] {local.name} -> {remote_path}")
        else:
            print(f"  [skip] {local} not found")

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nSpace is live: {space_url}")
    print("It may take 1-2 minutes to build. Check the 'App' tab.")


if __name__ == "__main__":
    main()
