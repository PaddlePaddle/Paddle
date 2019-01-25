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

from . import core
import numpy
import os
import six
from six.moves import zip, range, xrange
import multiprocessing

from .framework import Variable, default_main_program

__all__ = ['DataFeeder']


class DataToLoDTensorConverter(object):
    def __init__(self, place, lod_level, shape, dtype):
        self.place = place
        self.lod_level = lod_level
        self.shape = shape
        negtive_count = 0
        for s in self.shape:
            if s < 0:
                negtive_count += 1
            if negtive_count > 1:
                self.shape = None
                break
        if dtype == core.VarDesc.VarType.FP32:
            self.dtype = 'float32'
        elif dtype == core.VarDesc.VarType.INT64:
            self.dtype = 'int64'
        elif dtype == core.VarDesc.VarType.FP64:
            self.dtype = 'float64'
        elif dtype == core.VarDesc.VarType.FP16:
            self.dtype = 'float16'
        elif dtype == core.VarDesc.VarType.INT32:
            self.dtype = 'int32'
        elif dtype == core.VarDesc.VarType.UINT8:
            self.dtype = 'uint8'
        else:
            raise ValueError("dtype must be any of [int32, float32, int64, "
                             "float64, uint8]")

        self.data = []
        self.lod = []

        for i in six.moves.range(lod_level):
            self.lod.append([])

    def feed(self, data):
        self._feed_impl_(data, self.lod, self.lod_level)

    def _feed_impl_(self, data, lod, lod_level):
        if lod_level == 0:
            self.data.append(data)
        else:
            lod[0].append(len(data))
            for each_data in data:
                self._feed_impl_(each_data, lod[1:], lod_level - 1)

    def _check_shape(self, shape):
        for s1, s2 in zip(self.shape, shape):
            if s1 != s2 and s1 >= 0 and s2 >= 0:
                raise ValueError(
                    "Shape not match. What is defined in data layer is {}, but receive {}".
                    format(self.shape, shape))

    def done(self):
        arr = numpy.array(self.data, dtype=self.dtype)
        if self.shape:
            if len(arr.shape) != len(self.shape):
                try:
                    arr = arr.reshape(self.shape)
                except ValueError:
                    raise ValueError(
                        "Reshape error. What is defined in data layer is {}, but receive {}"
                        .format(self.shape, arr.shape))
            #else:
            #    self._check_shape(arr.shape)
        t = core.LoDTensor()
        t.set(arr, self.place)
        if self.lod_level > 0:
            t.set_recursive_sequence_lengths(self.lod)
        return t


class DataFeeder(object):
    """
    DataFeeder converts the data that returned by a reader into a data
    structure that can feed into Executor and ParallelExecutor. The reader
    usually returns a list of mini-batch data entries. Each data entry in
    the list is one sample. Each sample is a list or a tuple with one
    feature or multiple features.

    The simple usage shows below:

    ..  code-block:: python

        place = fluid.CPUPlace()
        img = fluid.layers.data(name='image', shape=[1, 28, 28])
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
        result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])


    If you want to feed data into GPU side separately in advance when you
    use multi-GPU to train a model, you can use `decorate_reader` function.

    ..  code-block:: python

        place=fluid.CUDAPlace(0)
        feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
        reader = feeder.decorate_reader(
            paddle.batch(flowers.train(), batch_size=16))

    Args:
        feed_list(list): The Variables or Variables'name that will
            feed into model.
        place(Place): place indicates feed data into CPU or GPU, if you want to
            feed data into GPU, please using `fluid.CUDAPlace(i)` (`i` represents
            the GPU id), or if you want to feed data into CPU, please using
            `fluid.CPUPlace()`.
        program(Program): The Program that will feed data into, if program
            is None, it will use default_main_program(). Default None.

    Raises:
        ValueError: If some Variable is not in this Program.

    Examples:
        .. code-block:: python

            # ...
            place = fluid.CPUPlace()
            feed_list = [
                main_program.global_block().var(var_name) for var_name in feed_vars_name
            ] # feed_vars_name is a list of variables' name.
            feeder = fluid.DataFeeder(feed_list, place)
            for data in reader():
                outs = exe.run(program=main_program,
                               feed=feeder.feed(data))
    """

    def __init__(self, feed_list, place, program=None):
        self.feed_dtypes = []
        self.feed_names = []
        self.feed_shapes = []
        self.feed_lod_level = []
        if program is None:
            program = default_main_program()
        for each_var in feed_list:
            if isinstance(each_var, six.string_types):
                each_var = program.block(0).var(each_var)
            if not isinstance(each_var, Variable):
                raise TypeError("Feed list should contain a list of variable")
            self.feed_dtypes.append(each_var.dtype)
            self.feed_names.append(each_var.name)
            self.feed_lod_level.append(each_var.lod_level)
            self.feed_shapes.append(each_var.shape)

        self.place = place

    def feed(self, iterable):
        """
        According to feed_list and iterable, converters the input into
        a data structure that can feed into Executor and ParallelExecutor.

        Args:
            iterable(list|tuple): the input data.

        Returns:
            dict: the result of conversion.
        """
        converter = []
        for lod_level, shape, dtype in six.moves.zip(
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
            for each_converter, each_slot in six.moves.zip(converter,
                                                           each_sample):
                each_converter.feed(each_slot)
        ret_dict = {}
        for each_name, each_converter in six.moves.zip(self.feed_names,
                                                       converter):
            ret_dict[each_name] = each_converter.done()
        return ret_dict

    def feed_parallel(self, iterable, num_places=None):
        """
        Takes multiple mini-batches. Each mini-batch will be feed on each
        device in advance.

        Args:
            iterable(list|tuple): the input data.
            num_places(int): the number of devices. Default None.

        Returns:
            dict: the result of conversion.

        Notes:
            The number of devices and number of mini-batches must be same.
        """
        if isinstance(self.place, core.CUDAPlace):
            places = [
                core.CUDAPlace(i)
                for i in six.moves.xrange(
                    self._get_number_of_places_(num_places))
            ]
        else:
            places = [
                core.CPUPlace()
                for _ in six.moves.xrange(
                    self._get_number_of_places_(num_places))
            ]

        if len(iterable) != len(places):
            raise ValueError("feed_parallel takes multiple mini-batches. Each "
                             "mini-batch will be feed on each device. The "
                             "number of devices and number of mini-batches "
                             "must be same.")

        place = self.place
        for p, batch in six.moves.zip(places, iterable):
            self.place = p
            yield self.feed(batch)
        self.place = place

    def _get_number_of_places_(self, num_places):
        if num_places is not None:
            return int(num_places)
        elif isinstance(self.place, core.CUDAPlace):
            return core.get_cuda_device_count()
        else:
            cpu_num = int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
            return cpu_num

    def decorate_reader(self,
                        reader,
                        multi_devices,
                        num_places=None,
                        drop_last=True):
        """
        Converter the input data into a data that returned by reader into
        multiple mini-batches. Each mini-batch will be feed on each device.

        Args:
            reader(function): the reader is the function which can generate data.
            multi_devices(bool): whether to use multiple devices or not.
            num_places(int): if the multi_devices is True, you can specify the number
                of GPU to use, if 'num_places' is None, the function will use all the
                GPU of the current machine. Default None.
            drop_last(bool): whether to drop the last batch if the
                size of the last batch is less than batch_size. Default True.

        Returns:
            dict: the result of conversion.

        Raises:
            ValueError: If drop_last is False and the data batch which cannot fit for devices.
        """

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
