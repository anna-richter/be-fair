import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/MyImages/image_252.jpg"

info = {}
info['file_path'] = file_path
info['file_size_kb'] = round(os.path.getsize(file_path)/1024, 2)

try:
    with Image.open(file_path) as img:
        info['format'] = img.format
        info['mode'] = img.mode
        info['size'] = img.size
except Exception as e:
    info['error'] = str(e)

for k, v in info.items():
    print(f"{k}: {v}")