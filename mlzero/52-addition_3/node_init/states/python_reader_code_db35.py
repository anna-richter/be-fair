import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data/MyImages/image_6056.jpg"

try:
    file_size = os.path.getsize(file_path) / (1024 * 1024)
except Exception as e:
    file_size = None

info = {}
info['file_path'] = file_path
info['file_size_MB'] = round(file_size, 2) if file_size else "Unknown"
info['file_type'] = "JPEG Image"

try:
    with Image.open(file_path) as img:
        info['format'] = img.format
        info['mode'] = img.mode
        info['size'] = img.size  # (width, height)
except Exception as e:
    info['error'] = str(e)

for k, v in info.items():
    print(f"{k}: {v}")