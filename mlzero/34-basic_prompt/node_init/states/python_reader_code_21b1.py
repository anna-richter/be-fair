import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/MyImages/image_4530.jpg"

info = {}
try:
    size_bytes = os.path.getsize(file_path)
    info['File Size (MB)'] = round(size_bytes / (1024*1024), 2)
    with Image.open(file_path) as img:
        info['Format'] = img.format
        info['Mode'] = img.mode
        info['Dimensions'] = img.size
except Exception as e:
    info['Error'] = str(e)

for k, v in info.items():
    print(f"{k}: {v}")