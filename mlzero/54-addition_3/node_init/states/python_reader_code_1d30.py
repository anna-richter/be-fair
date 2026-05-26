import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data/MyImages/image_5073.jpg"

try:
    file_size = os.path.getsize(file_path) / (1024 * 1024)
except Exception as e:
    file_size = None

info = {}
try:
    with Image.open(file_path) as img:
        info['format'] = img.format
        info['mode'] = img.mode
        info['size'] = img.size
except Exception as e:
    info['error'] = str(e)

print(f"File: {os.path.basename(file_path)}")
print(f"Type: Image (JPEG)")
print(f"Size: {file_size:.2f} MB")
for k, v in info.items():
    print(f"{k.capitalize()}: {v}")