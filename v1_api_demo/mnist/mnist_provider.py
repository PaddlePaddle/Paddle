from paddle.trainer.PyDataProvider2 import *
from mnist_util import read_from_mnist


# Define a py data provider
@provider(
    input_types={'pixel': dense_vector(28 * 28),
                 'label': integer_value(10)},
    cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, filename):  # settings is not used currently.
    for each in read_from_mnist(filename):
        yield each
