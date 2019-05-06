# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from . import core
import six
import threading
from .framework import Program, Variable, program_guard, default_main_program, default_startup_program
from .executor import global_scope
from .data_feeder import DataFeeder, BatchedTensorProvider
from .layers.io import monkey_patch_reader_methods, _copy_reader_var_, double_buffer
from .unique_name import UniqueNameGenerator

__all__ = ['PyReader']


def _convert_places(places):
    if not isinstance(places, (list, tuple)):
        places = [places]

    ret = []
    for p in places:
        if not isinstance(p, core.Place):
            tmp = core.Place()
            tmp.set_place(p)
            p = tmp

        ret.append(p)
    return ret


class PyReader(object):
    """
    Create a reader object for data feeding in Python. 
    Data would be prefetched using Python thread and be pushed
    into a queue asynchronously. Data in the queue would be extracted 
    automatically when `Executor.run(...)` is called.

    Args:  
        feed_list (list(Variable)|tuple(Variable)): feed variable list.
            The variables should be created by :code:`fluid.layers.data()`. 
        capacity (int): capacity of the queue maintained in PyReader object. 
        use_double_buffer (bool): whether to use double_buffer_reader to 
            speed up data feeding. 
        iterable (bool): whether the created reader object is iterable.   

    Returns:
        reader (Reader): the created reader object.

    Examples:
        1. If iterable = False, the created PyReader object is almost the
           same as :code:`fluid.layers.py_reader()`. Operators would be 
           inserted into the program. User should call :code:`start()` 
           before each epoch and catch :code:`fluid.core.EOFException`
           thrown by :code:`Executor.run()` when epoch ends. Once the 
           exception is caught, user should call :code:`reset()` to reset 
           the reader manually.

        .. code-block:: python

           EPOCH_NUM = 3
           ITER_NUM = 5

           def reader_creator_random_image_and_label(height, width):
               def reader():
                   for i in range(ITER_NUM):
                       fake_image = np.random.uniform(low=0,
                                                      high=255,
                                                      size=[height, width])
                       fake_label = np.ones([1])
                       fake_data = np.array([fake_image, fake_image])
                       yield np.expand_dims(fake_data, axis=0)
               return reader

           image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
           label = fluid.layers.data(name='label', shape=[1], dtype='int64')

           reader = fluid.io.PyReader(feed_list=[image, label],
                                      capacity=4,
                                      iterable=False)

           user_defined_reader = reader_creator_random_image_and_label(784, 784)
           reader.decorate_sample_list_generator(user_defined_reader)
           # definition of network is omitted
           executor = fluid.Executor(fluid.core.CUDAPlace(0))
           executor.run(fluid.default_startup_program())
           for i in range(EPOCH_NUM):
               reader.start()
               while True:
                   try:
                       executor.run(feed=None)
                   except fluid.core.EOFException:
                       reader.reset()
                       break 
                    
        2. If iterable=True, the created PyReader object is decoupled with
           the program. No operator would be inserted into the program. 
           In this case, the created reader is a Python generator, which 
           is iterable. User should feed the data yielded from PyReader 
           object into :code:`Executor.run(feed=...)`.  

        .. code-block:: python

           EPOCH_NUM = 3
           ITER_NUM = 5

           def reader_creator_random_image(height, width):
               def reader():
                   for i in range(ITER_NUM):
                       yield np.random.uniform(low=0,
                                               high=255,
                                               size=[1, 1, height, width])
               return reader

           image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')

           reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=True)

           user_defined_reader = reader_creator_random_image(784, 784)
           reader.decorate_sample_list_generator(user_defined_reader,
                                                 fluid.core.CUDAPlace(0))
           # definition of network is omitted
           executor = fluid.Executor(fluid.core.CUDAPlace(0))
           executor.run(fluid.default_main_program())

           for _ in range(EPOCH_NUM):
               for data in reader():
                   executor.run(feed=data)
    """

    unique_name_generator = UniqueNameGenerator()

    def __init__(self,
                 feed_list,
                 capacity,
                 use_double_buffer=True,
                 iterable=False):
        self._tensor_reader = None
        self._thread = None
        self._iterable = iterable
        self._use_double_buffer = use_double_buffer
        self._capacity = capacity
        self._feed_list = feed_list
        if not self._iterable:
            self._init_non_iterable()

    def _init_iterable(self, places):
        self._var_names = [v.name for v in self._feed_list]
        self._places = _convert_places(places)
        self._queue = core.init_lod_tensor_blocking_queue(core.Variable(),
                                                          self._capacity)
        self._reader = core.create_py_reader(
            self.queue, self._var_names, self._places, self._use_double_buffer)

    def _init_non_iterable(self):
        lod_levels = []
        dtypes = []
        shape_concat = []
        ranks = []
        shapes = []

        for feed_data in self._feed_list:
            dtypes.append(feed_data.dtype)
            shape_concat.extend(feed_data.shape)
            ranks.append(len(feed_data.shape))
            shapes.append(feed_data.shape)
            lod_levels.append(feed_data.lod_level)

        queue_name = PyReader.unique_name_generator('lod_tensor_blocking_queue')
        reader_name = PyReader.unique_name_generator('create_py_reader')
        double_buffer_name = PyReader.unique_name_generator('double_buffer')

        var = global_scope().var(queue_name)
        self._queue = core.init_lod_tensor_blocking_queue(var, self._capacity)

        startup_blk = default_startup_program().current_block()
        startup_var = startup_blk.create_var(name=reader_name)

        startup_blk.append_op(
            type='create_py_reader',
            inputs={'blocking_queue': [queue_name]},
            outputs={'Out': [startup_var]},
            attrs={
                'shape_concat': shape_concat,
                'lod_levels': lod_levels,
                'ranks': ranks
            })

        startup_var.desc.set_dtypes(dtypes)
        startup_var.persistable = True

        main_prog_var = _copy_reader_var_(
            default_main_program().current_block(), startup_var)

        main_prog_var.stop_gradient = True
        main_prog_var.persistable = True

        reader = monkey_patch_reader_methods(main_prog_var)
        if self._use_double_buffer:
            double_buffer_reader = double_buffer(
                reader, name=double_buffer_name)
            # we return a double buffer reader. However, the reset method comes from
            # py_reader.
            double_buffer_reader.reset = reader.reset
            reader = double_buffer_reader

        self._reader = reader

        default_main_program().current_block().append_op(
            type='read',
            inputs={'Reader': [self._reader]},
            outputs={'Out': self._feed_list})

    @property
    def queue(self):
        return self._queue

    @property
    def iterable(self):
        return self._iterable

    def __call__(self):
        assert self.iterable, "PyReader is not iterable"
        assert self._tensor_reader is not None, \
            "Data source of PyReader has not set yet"

        class Iterator(object):
            def __init__(self, reader):
                self._reader = reader._reader
                self._reset = reader._reset

            def __iter__(self):
                return self

            def __next__(self):
                return self.next()

            def next(self):
                ret = self._reader.read_next()
                if ret:
                    return ret
                else:
                    self._reset()
                    raise StopIteration

        self._start()
        return Iterator(self)

    def _reset(self):
        self._reader.reset()
        self._thread.join()

    def start(self):
        '''
        Start the data feeding thread. 
        Can only call when the reader object is not iterable.  
        
	Example:
	    .. code-block:: python
	        
                import paddle
                import paddle.fluid as fluid
                import paddle.dataset.mnist as mnist
                import numpy as np

                def generator():
                    for i in range(5):
		        yield np.random.uniform(low=0, high=255, size=[1, 1, 784, 784])

                image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')

                reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=False)
                reader.decorate_sample_list_generator(generator)

                executor = fluid.Executor(fluid.core.CUDAPlace(0))
                executor.run(fluid.default_startup_program())
                for i in range(3):
                    reader.start()
                    while True:
                        try:
                            executor.run(feed=None)
                        except fluid.core.EOFException:
                            reader.reset()
                            break
	'''
        assert not self._iterable, "start() cannot be called when PyReader is iterable"
        self._start()

    def reset(self):
        '''
        Reset the reader object when :code:`fluid.core.EOFException` raises. 
        Can only call when the reader object is not iterable.
        
        Example:
            .. code-block:: python
                
                import paddle
                import paddle.fluid as fluid
                import paddle.dataset.mnist as mnist
                import numpy as np

                def generator():
                    for i in range(5):
                        yield np.random.uniform(low=0, high=255, size=[1, 1, 784, 784])

                image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')

                reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=False)
                reader.decorate_sample_list_generator(generator)

                executor = fluid.Executor(fluid.core.CUDAPlace(0))
                executor.run(fluid.default_startup_program())
                for i in range(3):
                    reader.start()
                    while True:
                        try:
                            executor.run(feed=None)
                        except fluid.core.EOFException:
                            reader.reset()
                            break
        '''
        assert not self._iterable, "reset() cannot be called when PyReader is iterable"
        self._reset()

    def _start(self):
        def __thread_main__():
            try:
                for tensors in self._tensor_reader():
                    array = core.LoDTensorArray()
                    for item in tensors:
                        if not isinstance(item, core.LoDTensor):
                            tmp = core.LoDTensor()
                            tmp.set(item, core.CPUPlace())
                            item = tmp

                        array.append(item)

                    if not self._queue.push(array):
                        break

                self._queue.close()
            except Exception as ex:
                self._queue.close()
                raise ex

        self._thread = threading.Thread(target=__thread_main__)
        self._thread.daemon = True
        self._thread.start()

    def decorate_sample_generator(self,
                                  sample_generator,
                                  batch_size,
                                  drop_last=True,
                                  places=None):
        '''
        Set the data source of the PyReader object.
        
        The provided :code:`sample_generator` should be a Python generator,
        which yields numpy.ndarray typed data of each sample.

        :code:`places` must be set when the PyReader object is iterable.

        If all inputs have no lods, this method is faster than 
        :code:`decorate_sample_list_generator(paddle.batch(sample_generator, ...))` .

        Args:
            sample_generator (generator): Python generator that yields
                numpy.ndarray-typed sample data.
            batch_size (int): batch size. Must be larger than 0.
            drop_last (bool): Whether to drop the last batch when sample number
                is less than batch_size. 
            places (None|list(CUDAPlace)|list(CPUPlace)): place list. Must
                be provided when PyReader is iterable.

        Example:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                EPOCH_NUM = 3
                ITER_NUM = 5

                def random_image_generator(height, width):
                    def generator():
                        for i in range(ITER_NUM):
                            yield np.random.uniform(low=0, high=255, size=[1, height, width])
                    return generator

                image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
                reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=True)

                user_defined_generator = random_image_generator(784, 784)
                reader.decorate_sample_generator(user_defined_generator,
                                                 batch_size=8,
                                                 places=[fluid.core.CUDAPlace(0)])
                # definition of network is omitted
                executor = fluid.Executor(fluid.core.CUDAPlace(0))
                executor.run(fluid.default_main_program())

                for _ in range(EPOCH_NUM):
                    for data in reader():
                        executor.run(feed=data)
        '''
        assert batch_size > 0, "batch_size must be larger than 0"
        has_lod = False
        for f in self._feed_list:
            if f.lod_level != 0:
                has_lod = True
                break

        if has_lod:
            self.decorate_sample_list_generator(
                paddle.batch(
                    sample_generator,
                    batch_size=batch_size,
                    drop_last=drop_last),
                places=places)
        else:
            reader = BatchedTensorProvider(
                feed_list=self._feed_list,
                place=core.CPUPlace(),
                batch_size=batch_size,
                generator=sample_generator,
                drop_last=drop_last)
            self.decorate_batch_generator(reader, places=places)

    def decorate_sample_list_generator(self, reader, places=None):
        '''
        Set the data source of the PyReader object. 

        The provided :code:`reader` should be a Python generator,
        which yields list(numpy.ndarray) typed batched data. 
        
        :code:`places` must be set when the PyReader object is iterable.

        Args:
            reader (generator): Python generator that yields 
                list(numpy.ndarray)-typed batched data. 
            places (None|list(CUDAPlace)|list(CPUPlace)): place list. Must
                be provided when PyReader is iterable.
        
        Example:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                EPOCH_NUM = 3
                ITER_NUM = 5

                def random_image_generator(height, width):
                    def generator():
                        for i in range(ITER_NUM):
                            yield np.random.uniform(low=0,
                                                    high=255,
                                                    size=[1, 1, height, width])
                    return generator

                image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
                reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=True)

                user_defined_generator = random_image_generator(784, 784)
                reader.decorate_sample_list_generator(user_defined_generator,
                                                      fluid.core.CUDAPlace(0))
                # definition of network is omitted
                executor = fluid.Executor(fluid.core.CUDAPlace(0))
                executor.run(fluid.default_main_program())

                for _ in range(EPOCH_NUM):
                    for data in reader():
                        executor.run(feed=data)
        '''
        assert self._tensor_reader is None, \
            "Cannot reset the data source of PyReader"
        with program_guard(Program(), Program()):
            feeder = DataFeeder(
                feed_list=self._feed_list, place=core.CPUPlace())
            paddle_reader = feeder.decorate_reader(reader, multi_devices=False)

        def __tensor_reader_impl__():
            for slots in paddle_reader():
                yield [slots[var.name] for var in self._feed_list]

        self.decorate_batch_generator(__tensor_reader_impl__, places)

    def decorate_batch_generator(self, reader, places=None):
        '''
        Set the data source of the PyReader object.

        The provided :code:`reader` should be a Python generator,
        which yields numpy.ndarray-typed or LoDTensor-typed batched data.

        :code:`places` must be set when the PyReader object is iterable.

        Args:
            reader (generator): Python generator that yields LoDTensor-typed
                batched data.
            places (None|list(CUDAPlace)|list(CPUPlace)): place list. Must
                be provided when PyReader is iterable.

        Example:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                EPOCH_NUM = 3
                ITER_NUM = 5


                def random_image_generator(height, width):
                    def generator():
                        for i in range(ITER_NUM):
                            yield np.random.uniform(low=0,
                                                    high=255,
                                                    size=[1, 1, height, width])

                    return generator


                image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')

                reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=True)

                user_defined_generator = random_image_generator(784, 784)
                reader.decorate_batch_generator(user_defined_generator,
                                                fluid.core.CUDAPlace(0))
                # definition of network is omitted
                executor = fluid.Executor(fluid.core.CUDAPlace(0))
                executor.run(fluid.default_main_program())

                for _ in range(EPOCH_NUM):
                    for data in reader():
                        executor.run(feed=data)

        '''
        assert self._tensor_reader is None, \
            "Cannot reset the data source of PyReader"
        self._tensor_reader = reader
        if self._iterable:
            assert places is not None, "Places cannot be None when py_reader is iterable"
            self._init_iterable(places)
