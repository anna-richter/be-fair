import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/MyImages/image_13853.jpg"

try:
    file_size = os.path.getsize(file_path) / (1024 * 1024)
except Exception as e:
    file_size = None

info = {}
info['File'] = os.path.basename(file_path)
info['Size_MB'] = round(file_size, 3) if file_size else "Unknown"
info['Type'] = "Unknown"

try:
    with Image.open(file_path) as img:
        info['Type'] = img.format
        info['Mode'] = img.mode
        info['Size'] = img.size
except Exception as e:
    info['Error'] = str(e)

for k, v in info.items():
    print(f"{k}: {v}")