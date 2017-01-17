from paddle.trainer.PyDataProvider2 import *
from py_paddle.dataprovider_converter import DataProviderConverter

__all__ = [
    'dense_vector', 'dense_vector_sequence', 'dense_vector_sub_sequence',
    'integer_value', 'integer_sequence', 'integer_value_sub_sequence',
    'sparse_binary_vector', 'sparse_binary_vector_sequence',
    'sparse_binary_vector_sub_sequence', 'sparse_vector',
    'sparse_vector_sequence', 'sparse_vector_sub_sequence', 'provider',
    'CacheType', 'DataProviderConverter', 'chunk', 'IDataPool'
]


def chunk(iterable, size=1):
    items = [None] * size  # prealloc
    for i, item in enumerate(iterable):
        if i % size == 0 and i != 0:
            yield items
        items[i % size] = item
    i += 1  # i is the total size.
    i %= size
    if i == 0:
        yield items
    else:
        yield items[:min(i + 1, size)]


class IDataPool(object):
    """
    Interface of DataPool, but note that Python is using Duck-Typing, it is not
    necessary to inherit this interface.

    NOTE: For Paddle developer, NEVER CHECK isinstance(obj, IDataPool).

    Basically contains two method,

    * next(): User should return the next batch of data in pool. raise
              StopIteration if there is no more data in pool.

    * reset(): Reset the data pool to initial status.

    The basic usage of this api is as same as normal Python iterator, like

    ..  code-block:: python

        pool = DataPool()

        for batch in pool:
            process_batch(batch)


    NOTE: The Data Pool API is not thread-safe.
    """

    def __iter__(self):
        self.reset()
        return self

    def next(self):
        raise NotImplementedError()

    def __next__(self):
        return self.next()

    def reset(self):
        raise NotImplementedError()
