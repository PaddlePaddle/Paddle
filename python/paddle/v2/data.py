import collections
import random

__all__ = [
    'IDataPool',
    'CacheAllDataPool',
]


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


def input_order_mapper(iterable, input_order):
    assert isinstance(input_order, collections.Sequence)
    for each_input_name in input_order:
        assert isinstance(each_input_name, basestring)

    tmp = [None] * len(input_order)
    for each_item in iterable:
        for i in xrange(len(input_order)):
            tmp[i] = each_item[input_order[i]]
        yield tmp


class CacheAllDataPool(IDataPool):
    """
    load all samples in memory.
    """

    def __init__(self, iterable, batch_size, input_order, shuffle=True):
        self.__pool__ = list(
            input_order_mapper(
                iterable=iterable, input_order=input_order))
        self.__batch_size__ = batch_size
        self.__shuffle__ = shuffle
        self.__idx__ = 0

    def reset(self):
        self.__idx__ = 0
        if self.__shuffle__:
            random.shuffle(self.__pool__)

    def next(self):
        if self.__idx__ >= len(self.__pool__):
            raise StopIteration()

        begin = self.__idx__
        end = min(self.__idx__ + self.__batch_size__, len(self.__pool__))
        self.__idx__ = end
        return self.__pool__[begin:end]
