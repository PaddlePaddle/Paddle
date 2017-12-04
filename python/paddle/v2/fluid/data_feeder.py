from __future__ import print_function
import core
import numpy
import six.moves as six

from framework import Variable

__all__ = ['DataFeeder']


class DataToLoDTensorConverter(object):
    def __init__(self, place, lod_level, shape, dtype):
        self.place = place
        self.lod_level = lod_level
        self.shape = shape
        if dtype == core.DataType.FP32:
            self.dtype = 'float32'
        elif dtype == core.DataType.INT64:
            self.dtype = 'int64'
        elif dtype == core.DataType.FP64:
            self.dtype = 'float64'
        elif dtype == core.DataType.INT32:
            self.dtype = 'int32'
        else:
            raise ValueError("dtype must be any of [int32, float32, int64, "
                             "float64]")

        self.data = []
        self.lod = []

        for i in six.range(lod_level):
            self.lod.append([0])

    def feed(self, data):
        self._feed_impl_(data, self.lod, self.lod_level)

    def _feed_impl_(self, data, lod, lod_level):
        if lod_level == 0:
            self.data.append(data)
        else:
            cur_lod_len = len(data)
            lod[-1].append(lod[-1][-1] + cur_lod_len)
            for each_data in data:
                self._feed_impl_(each_data, lod[:-1], lod_level - 1)

    def done(self):
        arr = numpy.array(self.data, dtype=self.dtype).reshape(self.shape)
        t = core.LoDTensor()
        t.set(arr, self.place)
        if self.lod_level > 0:
            t.set_lod(self.lod)
        return t


class DataFeeder(object):
    def __init__(self, feed_list, place):
        self.feed_dtypes = []
        self.feed_names = []
        self.feed_shapes = []
        self.feed_lod_level = []
        for each_var in feed_list:
            if not isinstance(each_var, Variable):
                raise TypeError("Feed list should contain a list of variable")
            self.feed_dtypes.append(each_var.dtype)
            self.feed_names.append(each_var.name)
            shape = each_var.shape
            batch_size_dim = -1
            for i, s in enumerate(shape):
                if s < 0:
                    batch_size_dim = i
                    break
            if batch_size_dim == -1:
                raise ValueError("Variable {0} must has a batch size dimension",
                                 each_var.name)
            self.feed_lod_level.append(each_var.lod_level)
            self.feed_shapes.append(shape)

        self.place = place

    def feed(self, iterable):
        converter = []
        for lod_level, shape, dtype in six.zip(
                self.feed_lod_level, self.feed_shapes, self.feed_dtypes):
            converter.append(
                DataToLoDTensorConverter(
                    place=self.place,
                    lod_level=lod_level,
                    shape=shape,
                    dtype=dtype))

        for each_sample in iterable:
            for each_converter, each_slot in six.zip(converter, each_sample):
                each_converter.feed(each_slot)
        ret_dict = {}
        for each_name, each_converter in six.zip(self.feed_names, converter):
            ret_dict[each_name] = each_converter.done()
        return ret_dict
