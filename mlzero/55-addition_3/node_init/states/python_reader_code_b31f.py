import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data/MyImages/image_3862.jpg"

try:
    file_size = os.path.getsize(file_path) / (1024 * 1024)
except Exception as e:
    file_size = None

info = []
info.append(f"File: {os.path.basename(file_path)}")
if file_size is not None:
    info.append(f"Size: {file_size:.2f} MB")

try:
    with Image.open(file_path) as img:
        info.append(f"Type: Image ({img.format})")
        info.append(f"Mode: {img.mode}")
        info.append(f"Size: {img.size[0]}x{img.size[1]}")
except Exception as e:
    info.append("Type: Unknown or unreadable file.")

print('\n'.join(info))