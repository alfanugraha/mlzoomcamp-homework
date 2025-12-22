# IMPORTS
from io import BytesIO
from urllib import request
from PIL import Image

import onnxruntime as ort
import numpy as np

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

session = ort.InferenceSession("hair_classifier_empty.onnx")

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name

img_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
img = download_image(img_url)
img_rz = prepare_image(img, (200, 200))

x = np.array(img_rz) 
x = x / 255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
x = (x - mean) / std

print(f"The value of first pixel in the R channel: {x[0][0][0]}")

x = np.transpose(x, (2, 0, 1))  
x = np.expand_dims(x, axis=0)  
x = x.astype(np.float32)

# Run inference
res = session.run([output_name], {input_name: x})[0]
print("Clothing probability: ", res[0][0])
