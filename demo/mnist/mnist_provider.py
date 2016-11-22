from paddle.trainer.PyDataProvider2 import *


# Define a py data provider
@provider(input_types={
    'pixel': dense_vector(28 * 28),
    'label': integer_value(10)
})
def process(settings, filename):  # settings is not used currently.
    imgf = filename + "-images-idx3-ubyte"
    labelf = filename + "-labels-idx1-ubyte"
    f = open(imgf, "rb")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)

    # Define number of samples for train/test
    if "train" in filename:
        n = 60000
    else:
        n = 10000

    for i in range(n):
        label = ord(l.read(1))
        pixels = []
        for j in range(28 * 28):
            pixels.append(float(ord(f.read(1))) / 255.0)
        yield {"pixel": pixels, 'label': label}

    f.close()
    l.close()
