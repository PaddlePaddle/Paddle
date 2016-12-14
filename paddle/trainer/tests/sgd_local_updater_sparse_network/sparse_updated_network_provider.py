from paddle.trainer.PyDataProvider2 import *
import random


@provider(
    input_types={"word_id": integer_value(600000),
                 "label": integer_value(10)},
    min_pool_size=0)
def process(settings, filename):
    for _ in xrange(1000):
        yield random.randint(0, 600000 - 1), random.randint(0, 9)
