from paddle.trainer.PyDataProvider2 import *
import numpy


# Define a py data provider
@provider(
    input_types={'pixel': dense_vector(28 * 28),
                 'label': integer_value(10)},
    cache=CacheType.CACHE_PASS_IN_MEM)
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

    images = numpy.fromfile(
        f, 'ubyte', count=n * 28 * 28).reshape((n, 28 * 28)).astype('float32')
    images = images / 255.0 * 2.0 - 1.0
    labels = numpy.fromfile(l, 'ubyte', count=n).astype("int")

    for i in xrange(n):
        yield {"pixel": images[i, :], 'label': labels[i]}

    f.close()
    l.close()
