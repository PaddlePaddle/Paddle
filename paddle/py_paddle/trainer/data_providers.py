from .. import DataProviderConverter
import random

__all__ = ['DataProvider', 'NaiveDataProvider']


class DataProvider(object):
    __slots__ = [
        '__init__', 'reset', 'next', '__provider__', '__converter__',
        '__batch_size__', '__should_shuffle__'
    ]

    def __init__(self, provider, input_types, batch_size, should_shuffle=True):
        self.__provider__ = provider
        self.__converter__ = DataProviderConverter(input_types)
        self.__batch_size__ = batch_size
        if self.__provider__.should_shuffle is None:
            self.__provider__.should_shuffle = should_shuffle

    def reset(self):
        raise NotImplemented()

    def next(self):
        raise NotImplemented()

    def __should_shuffle__(self):
        return self.__provider__.should_shuffle


class NaiveDataProvider(DataProvider):
    def __init__(self, provider, input_types, batch_size, should_shuffle=True):
        super(NaiveDataProvider, self).__init__(
            provider=provider,
            input_types=input_types,
            batch_size=batch_size,
            should_shuffle=should_shuffle)
        self.__pool__ = []
        self.__idx__ = 0

    def reset(self):
        def __to_pool__():
            for filename in self.__provider__.file_list:
                for item in self.__provider__.generator(self.__provider__,
                                                        filename):
                    yield item

        self.__pool__ = list(__to_pool__())
        if self.__should_shuffle__():
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
