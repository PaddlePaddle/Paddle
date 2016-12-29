from .. import DataProviderConverter
import random

__all__ = ['DataProvider', 'NaiveMemPooledDataProvider', 'NaiveDataProvider']


class DataProvider(object):
    __slots__ = [
        '__init__', 'reset', 'next', '__method__', '__converter__',
        '__batch_size__', '__should_shuffle__'
    ]

    def __init__(self, method, input_types, batch_size, should_shuffle=True):
        self.__method__ = method
        self.__converter__ = DataProviderConverter(input_types)
        self.__batch_size__ = batch_size
        self.__should_shuffle__ = should_shuffle

    def reset(self):
        raise NotImplemented()

    def next(self):
        raise NotImplemented()


class NaiveMemPooledDataProvider(DataProvider):
    def __init__(self, method, input_types, batch_size, should_shuffle):
        super(NaiveMemPooledDataProvider, self).__init__(
            method=method,
            input_types=input_types,
            batch_size=batch_size,
            should_shuffle=should_shuffle)
        self.__pool__ = []
        self.__idx__ = 0

    def reset(self):
        self.__pool__ = list(self.__method__())
        if self.__should_shuffle__:
            random.shuffle(self.__pool__)

        self.__idx__ = 0

    def next(self):
        if self.__idx__ < len(self.__pool__):
            end = min(self.__idx__ + self.__batch_size__, len(self.__pool__))
            begin = self.__idx__
            self.__idx__ = end
            return self.__converter__(self.__pool__[begin:end]), end - begin
        else:
            raise StopIteration


class NaiveDataProvider(NaiveMemPooledDataProvider):
    def __init__(self, provider, input_types, batch_size, should_shuffle=True):
        def __to_pool__():
            for filename in provider.file_list:
                for item in provider.generator(provider, filename):
                    yield item

        super(NaiveDataProvider, self).__init__(
            method=__to_pool__,
            input_types=input_types,
            batch_size=batch_size,
            should_shuffle=should_shuffle)
