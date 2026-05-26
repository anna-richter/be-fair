import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/MyImages/image_3623.jpg"

def analyze_image(path):
    try:
        with Image.open(path) as img:
            info = {
                "File": os.path.basename(path),
                "Format": img.format,
                "Mode": img.mode,
                "Size (WxH)": img.size,
                "File Size (KB)": round(os.path.getsize(path)/1024, 2)
            }
            for k, v in info.items():
                print(f"{k}: {v}")
    except Exception as e:
        print(f"Could not open image: {e}")

analyze_image(file_path)