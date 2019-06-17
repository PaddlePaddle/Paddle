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

from .framework import Variable, default_main_program, _current_expected_place
from .framework import _cpu_num, _cuda_ids
__all__ = ['DataFeeder']


def convert_dtype(dtype):
    if dtype == core.VarDesc.VarType.FP32:
        return 'float32'
    elif dtype == core.VarDesc.VarType.INT64:
        return 'int64'
    elif dtype == core.VarDesc.VarType.FP64:
        return 'float64'
    elif dtype == core.VarDesc.VarType.FP16:
        return 'float16'
    elif dtype == core.VarDesc.VarType.INT32:
        return 'int32'
    elif dtype == core.VarDesc.VarType.UINT8:
        return 'uint8'
    else:
        raise ValueError("dtype must be any of [int32, float32, int64, "
                         "float64, uint8]")


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
        self.dtype = convert_dtype(dtype)
        self._reset()

    def _reset(self):
        self.data = []
        self.lod = [[] for _ in six.moves.range(self.lod_level)]

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
        t = core.LoDTensor()
        t.set(arr, self.place)
        if self.lod_level > 0:
            t.set_recursive_sequence_lengths(self.lod)
        self._reset()
        return t


class BatchedTensorProvider(object):
    def __init__(self, feed_list, place, batch_size, generator, drop_last):
        self.place = place
        self.batch_size = batch_size
        self.generator = generator
        self.converters = []
        self.drop_last = drop_last

        for var in feed_list:
            assert var.lod_level == 0, "lod_level must be 0"
            self.converters.append(
                DataToLoDTensorConverter(
                    place=self.place,
                    lod_level=0,
                    shape=var.shape,
                    dtype=var.dtype))

    def _done(self):
        return [c.done() for c in self.converters]

    def __call__(self):
        idx = 0
        for each_sample in self.generator():
            for each_slot, each_converter in six.moves.zip(each_sample,
                                                           self.converters):
                each_converter.data.append(each_slot)

            idx += 1
            if idx == self.batch_size:
                idx = 0
                yield self._done()

        if not self.drop_last and idx > 0:
            yield self._done()
        else:
            [c._reset() for c in self.converters]


class DataFeeder(object):
    """
    DataFeeder converts the data that returned by a reader into a data
    structure that can feed into Executor and ParallelExecutor. The reader
    usually returns a list of mini-batch data entries. Each data entry in
    the list is one sample. Each sample is a list or a tuple with one
    feature or multiple features.

    The simple usage shows below:

    ..  code-block:: python

        import paddle.fluid as fluid
        place = fluid.CPUPlace()
        img = fluid.layers.data(name='image', shape=[1, 28, 28])
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
        result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])


    If you want to feed data into GPU side separately in advance when you
    use multi-GPU to train a model, you can use `decorate_reader` function.

    ..  code-block:: python

        import paddle
        import paddle.fluid as fluid
        
        place=fluid.CUDAPlace(0)
        data = fluid.layers.data(name='data', shape=[3, 224, 224], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        
        feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
        reader = feeder.decorate_reader(
                paddle.batch(paddle.dataset.flowers.train(), batch_size=16), multi_devices=False)

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
        ..  code-block:: python


            import numpy as np
            import paddle
            import paddle.fluid as fluid
            
            place = fluid.CPUPlace()
            
            def reader():
                yield [np.random.random([4]).astype('float32'), np.random.random([3]).astype('float32')],
            
            main_program = fluid.Program()
            startup_program = fluid.Program()
            
            with fluid.program_guard(main_program, startup_program):
                data_1 = fluid.layers.data(name='data_1', shape=[1, 2, 2])
                data_2 = fluid.layers.data(name='data_2', shape=[1, 1, 3])
                out = fluid.layers.fc(input=[data_1, data_2], size=2)
                # ...
            
            feeder = fluid.DataFeeder([data_1, data_2], place)
                        
            exe = fluid.Executor(place)
            exe.run(startup_program)
            for data in reader():
                outs = exe.run(program=main_program,
                               feed=feeder.feed(data),
                               fetch_list=[out])

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

        Examples:
            ..  code-block:: python

                import numpy.random as random
                import paddle.fluid as fluid
                
                def reader(limit=5):
                    for i in range(limit):
                        yield random.random([784]).astype('float32'), random.random([1]).astype('int64'), random.random([256]).astype('float32')
                
                data_1 = fluid.layers.data(name='data_1', shape=[1, 28, 28])
                data_2 = fluid.layers.data(name='data_2', shape=[1], dtype='int64')
                data_3 = fluid.layers.data(name='data_3', shape=[16, 16], dtype='float32')
                feeder = fluid.DataFeeder(['data_1','data_2', 'data_3'], fluid.CPUPlace())
                
                result = feeder.feed(reader()) 
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

        Examples:
            ..  code-block:: python

                import numpy.random as random
                import paddle.fluid as fluid
                
                def reader(limit=10):
                    for i in range(limit):
                        yield [random.random([784]).astype('float32'), random.randint(10)],
                
                x = fluid.layers.data(name='x', shape=[1, 28, 28])
                y = fluid.layers.data(name='y', shape=[1], dtype='int64')
                
                feeder = fluid.DataFeeder(['x','y'], fluid.CPUPlace())
                place_num = 2
                places = [fluid.CPUPlace() for x in range(place_num)]
                data = []
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(fluid.default_startup_program())
                program = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(places=places)
                for item in reader():
                    data.append(item)
                    if place_num == len(data):
                        exe.run(program=program, feed=list(feeder.feed_parallel(data, place_num)), fetch_list=[])
                        data = []
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
            return len(_cuda_ids())
        else:
            return _cpu_num()

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
            num_places(int): if multi_devices is True, you can specify the number
                of GPU to use, if multi_devices is None, the function will use all the
                GPU of the current machine. Default None.
            drop_last(bool): whether to drop the last batch if the
                size of the last batch is less than batch_size. Default True.

        Returns:
            dict: the result of conversion.

        Raises:
            ValueError: If drop_last is False and the data batch cannot fit for devices.

        Examples:
            ..  code-block:: python

                import numpy.random as random
                import paddle
                import paddle.fluid as fluid
                
                def reader(limit=5):
                    for i in range(limit):
                        yield (random.random([784]).astype('float32'), random.random([1]).astype('int64')),
                
                place=fluid.CUDAPlace(0)
                data = fluid.layers.data(name='data', shape=[1, 28, 28], dtype='float32')
                label = fluid.layers.data(name='label', shape=[1], dtype='int64')
                
                feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
                reader = feeder.decorate_reader(reader, multi_devices=False)
                
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in reader():
                    exe.run(feed=data)
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


class NumpyToLoDTensorConverter(object):
    def __init__(self, place):
        self.place = place
        self.data = []
        self._reset()

    def _reset(self):
        self.data = []

    def feed(self, data):
        self.data.append(data)

    def done(self):
        arr = numpy.array(self.data)
        t = core.LoDTensor()
        t.set(arr, self.place)
        self._reset()
        return t


class ListTensorProvider(object):
    def __init__(self, generator, places):
        self.generator = generator
        self.converters = []
        self.places = []
        if places:
            if not isinstance(places, (list, tuple)):
                places = [places]
            assert len(
                places) == 1, "dygraph mode CAN NOT specify multiple places."
            for place in places:
                if isinstance(place, (core.CUDAPlace, core.CPUPlace)):
                    self.places.append(place)
                else:
                    raise ValueError(
                        "Please specify a valid place values such as core.CPUPlace or core.CUDAPlace"
                    )
        if len(self.places) == 0:
            self.places.append(_current_expected_place())

    def _readData(self, iterable, places):
        for place, each_sample in six.moves.zip(places, iterable):
            for item in each_sample:
                if len(self.converters) < len(item):
                    for i in item:
                        self.converters.append(NumpyToLoDTensorConverter(place))
                for each_converter, each_slot in six.moves.zip(self.converters,
                                                               item):
                    each_converter.feed(each_slot)
            yield [c.done() for c in self.converters]

    def __call__(self):
        item = []
        for batch in self.generator():
            item.append(batch)
            if len(item) == len(self.places):
                yield list(self._readData(item, self.places))
                item = []
