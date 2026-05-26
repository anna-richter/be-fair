import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data/MyImages/image_2039.jpg"

def analyze_image(path):
    try:
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes / (1024 * 1024)
        with Image.open(path) as img:
            info = {
                "File": os.path.basename(path),
                "Type": img.format,
                "Mode": img.mode,
                "Size (pixels)": img.size,
                "File Size (MB)": round(size_mb, 3)
            }
        for k, v in info.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"Could not analyze image: {e}")

analyze_image(file_path)