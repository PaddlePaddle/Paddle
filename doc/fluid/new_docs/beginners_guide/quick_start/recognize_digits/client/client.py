import requests
from PIL import Image
import numpy as np
import os

# this client is used by Paddle serve: https://github.com/PaddlePaddle/book/tree/develop/serve
# please do not use it directly


def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32).flatten()
    im = im / 255.0
    return im


cur_dir = os.path.dirname(os.path.realpath(__file__))
data = load_image(cur_dir + '/../image/infer_3.png')
data = data.tolist()

r = requests.post("http://0.0.0.0:8000", json={'img': data})

print(r.text)
