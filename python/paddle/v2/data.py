__all__ = ['IData', ]


class IDataIter(object):
    def __iter__(self):
        self.reset()
        return self

    def next(self):
        raise NotImplementedError()

    def __next__(self):
        return self.next()

    def reset(self):
        raise NotImplementedError()
