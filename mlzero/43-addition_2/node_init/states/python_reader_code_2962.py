import os
from PIL import Image

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data/MyImages/image_3748.jpg"

def analyze_image(path):
    info = {}
    info['File'] = os.path.basename(path)
    info['Size_MB'] = round(os.path.getsize(path) / (1024*1024), 2)
    try:
        with Image.open(path) as img:
            info['Format'] = img.format
            info['Mode'] = img.mode
            info['Dimensions'] = img.size
    except Exception as e:
        info['Error'] = str(e)
    for k, v in info.items():
        print(f"{k}: {v}")

analyze_image(file_path)