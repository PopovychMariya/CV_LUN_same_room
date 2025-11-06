import os
import sys
import zipfile
from pathlib import Path
from path_config import ARCHIVES_PATH


def make_archive(folder_path, archive_name):
    folder_path = Path(folder_path).resolve()
    ARCHIVES_PATH.mkdir(parents=True, exist_ok=True)
    archive_name = str(archive_name)
    if not archive_name.endswith(".zip"):
        archive_name += ".zip"
    archive_path = ARCHIVES_PATH / archive_name

    top = folder_path.name

    with zipfile.ZipFile(str(archive_path), "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.startswith(".") or file.endswith((".DS_Store", "Thumbs.db")):
                    continue
                fp = Path(root) / file
                rel = os.path.relpath(fp, start=folder_path)
                arcname = os.path.join(top, rel)
                zipf.write(str(fp), arcname)

    print(f"Archive created: {archive_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python archive_folder.py <folder_path> <archive_name>")
        sys.exit(1)
    make_archive(sys.argv[1], sys.argv[2])