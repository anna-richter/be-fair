import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/MyImages/image_4982.jpg"

try:
    file_size = os.path.getsize(file_path) / (1024 * 1024)
except Exception as e:
    file_size = None

info = {"File": file_path, "Type": "Unknown", "Size_MB": round(file_size, 4) if file_size else "N/A"}

try:
    with Image.open(file_path) as img:
        info["Type"] = f"Image ({img.format})"
        info["Mode"] = img.mode
        info["Size_px"] = img.size
except Exception as e:
    info["Error"] = str(e)

for k, v in info.items():
    print(f"{k}: {v}")