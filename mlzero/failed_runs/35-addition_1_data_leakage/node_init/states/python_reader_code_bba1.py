import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/MyImages/image_12560.jpg"

def analyze_image(path):
    info = {}
    try:
        with Image.open(path) as img:
            info['format'] = img.format
            info['mode'] = img.mode
            info['size'] = img.size
    except Exception as e:
        info['error'] = str(e)
    return info

file_size = os.path.getsize(file_path) / (1024 * 1024)
info = analyze_image(file_path)

print(f"File: {os.path.basename(file_path)}")
print(f"Type: JPEG Image")
print(f"Size: {file_size:.2f} MB")
if 'error' in info:
    print(f"Error reading image: {info['error']}")
else:
    print(f"Image format: {info['format']}")
    print(f"Image mode: {info['mode']}")
    print(f"Image size (WxH): {info['size']}")