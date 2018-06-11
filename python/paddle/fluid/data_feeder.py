#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import core
import numpy
import six.moves as six
import multiprocessing

from framework import Variable, default_main_program

__all__ = ['DataFeeder']


class DataToLoDTensorConverter(object):
    def __init__(self, place, lod_level, shape, dtype):
        self.place = place
        self.lod_level = lod_level
        self.shape = shape
        if dtype == core.VarDesc.VarType.FP32:
            self.dtype = 'float32'
        elif dtype == core.VarDesc.VarType.INT64:
            self.dtype = 'int64'
        elif dtype == core.VarDesc.VarType.FP64:
            self.dtype = 'float64'
        elif dtype == core.VarDesc.VarType.INT32:
            self.dtype = 'int32'
        elif dtype == core.VarDesc.VarType.UINT8:
            self.dtype = 'uint8'
        else:
            raise ValueError("dtype must be any of [int32, float32, int64, "
                             "float64, uint8]")

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
            lod[0].append(lod[0][-1] + cur_lod_len)
            for each_data in data:
                self._feed_impl_(each_data, lod[1:], lod_level - 1)

    def done(self):
        arr = numpy.array(self.data, dtype=self.dtype).reshape(self.shape)
        t = core.LoDTensor()
        t.set(arr, self.place)
        if self.lod_level > 0:
            t.set_lod(self.lod)
        return t


class DataFeeder(object):
    def __init__(self, feed_list, place, program=None):
        self.feed_dtypes = []
        self.feed_names = []
        self.feed_shapes = []
        self.feed_lod_level = []
        if program is None:
            program = default_main_program()
        for each_var in feed_list:
            if isinstance(each_var, basestring):
                each_var = program.block(0).var(each_var)
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
            assert len(each_sample) == len(converter), (
                "The number of fields in data (%s) does not match " +
                "len(feed_list) (%s)") % (len(each_sample), len(converter))
            for each_converter, each_slot in six.zip(converter, each_sample):
                each_converter.feed(each_slot)
        ret_dict = {}
        for each_name, each_converter in six.zip(self.feed_names, converter):
            ret_dict[each_name] = each_converter.done()
        return ret_dict

    def feed_parallel(self, iterable, num_places=None):
        if isinstance(self.place, core.CUDAPlace):
            places = [
                core.CUDAPlace(i)
                for i in six.xrange(self._get_number_of_places_(num_places))
            ]
        else:
            places = [
                core.CPUPlace()
                for _ in six.xrange(self._get_number_of_places_(num_places))
            ]

        if len(iterable) != len(places):
            raise ValueError("feed_parallel takes multiple mini-batches. Each "
                             "mini-batch will be feed on each device. The "
                             "number of devices and number of mini-batches "
                             "must be same.")

        place = self.place
        for p, batch in six.zip(places, iterable):
            self.place = p
            yield self.feed(batch)
        self.place = place

    def _get_number_of_places_(self, num_places):
        if num_places is not None:
            return int(num_places)
        elif isinstance(self.place, core.CUDAPlace):
            return core.get_cuda_device_count()
        else:
            return multiprocessing.cpu_count()

    def decorate_reader(self,
                        reader,
                        multi_devices,
                        num_places=None,
                        drop_last=True):
        def __reader_creator__():
            if not multi_devices:
                for item in reader():
                    yield self.feed(item)
            else:
                num = self._get_number_of_places_(num_places)
                item = []
                for batch in reader():
                    item.append(batch)
                    if len(item) == num:
                        yield list(self.feed_parallel(item, num))
                        item = []
                if not drop_last and len(item) != 0:
                    raise ValueError(
                        "The data batch which cannot fit for devices will be "
                        "dropped is not implementation. Other strategies are "
                        "not implemented")

        return __reader_creator__
