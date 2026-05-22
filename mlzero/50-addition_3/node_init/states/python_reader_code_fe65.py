import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data/MyImages/image_3701.jpg"

def analyze_image(path):
    try:
        size_bytes = os.path.getsize(path)
        with Image.open(path) as img:
            info = {
                "File": os.path.basename(path),
                "Format": img.format,
                "Mode": img.mode,
                "Size (pixels)": img.size,
                "File Size (KB)": round(size_bytes / 1024, 2)
            }
        for k, v in info.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"Could not analyze image: {e}")

analyze_image(file_path)