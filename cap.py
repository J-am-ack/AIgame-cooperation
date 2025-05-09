import os
from pathlib import Path

images_dir = Path("posterset/images")
captions_dir = Path("posterset/captions")
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


for img_path in images_dir.glob("*"):
    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
        txt_path = captions_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():
            txt_path.write_text("", encoding="utf-8")
            print(f"已创建空白标注文件: {txt_path.name}")
        else:
            print(f"已存在标注文件: {txt_path.name}")

