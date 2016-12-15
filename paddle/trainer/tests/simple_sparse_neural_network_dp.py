from paddle.trainer.PyDataProvider2 import provider, integer_sequence, integer_value
import random


def init_hook(settings, is_train, **kwargs):
    settings.is_train = is_train


@provider(
    input_types={'word_ids': integer_value(65536),
                 'label': integer_value(10)},
    min_pool_size=0,
    init_hook=init_hook)
def process(settings, filename):
    if settings.is_train:
        data_size = 2**20
    else:
        data_size = 2**10

    for _ in xrange(data_size):
        yield random.randint(0, 65535), random.randint(0, 9)
