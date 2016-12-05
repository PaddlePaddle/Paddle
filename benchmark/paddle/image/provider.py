import io, os
import random
import numpy as np
from paddle.trainer.PyDataProvider2 import *


def initHook(settings, height, width, color, num_class, **kwargs):
    settings.height = height
    settings.width = width
    settings.color = color
    settings.num_class = num_class
    if settings.color:
        settings.data_size = settings.height * settings.width * 3
    else:
        settings.data_size = settings.height * settings.width

    settings.slots = [dense_vector(settings.data_size), integer_value(1)]


@provider(
    init_hook=initHook, min_pool_size=-1, cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, file_list):
    for i in xrange(1024):
        img = np.random.rand(1, settings.data_size).reshape(-1, 1).flatten()
        lab = random.randint(0, settings.num_class)
        yield img.astype('float32'), int(lab)
