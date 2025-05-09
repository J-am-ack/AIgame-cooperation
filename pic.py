import os
from pathlib import Path

images_dir = Path("新形象合集")
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp",
                    ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

for i, img in enumerate(images_dir.glob("*")):
    if img.suffix.lower() in IMAGE_EXTENSIONS:
        new_name = f"{i:04d}{img.suffix}"
        img.rename(images_dir / new_name)