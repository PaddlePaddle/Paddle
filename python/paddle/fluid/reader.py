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

from . import core, dygraph
import sys
import six
import warnings
import numpy as np
import threading
import paddle
from .framework import Program, Variable, program_guard, default_main_program, default_startup_program, in_dygraph_mode
from .executor import global_scope
from .data_feeder import DataFeeder, BatchedTensorProvider, ListTensorProvider
from .layers.io import monkey_patch_reader_methods, _copy_reader_var_, double_buffer
from .unique_name import UniqueNameGenerator
import logging
import time

__all__ = ['PyReader']

data_loader_unique_name_generator = UniqueNameGenerator()


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


class DataLoader(object):
    def __init__(self):
        self._places = None

    def __call__(self):
        return self

    def next(self):
        return self.__next__()

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()


class DataLoaderFactory(object):
    @staticmethod
    def from_generator(feed_list=None,
                       capacity=None,
                       use_double_buffer=True,
                       iterable=True,
                       return_list=False):
        return GeneratorLoader(feed_list, capacity, use_double_buffer, iterable,
                               return_list)

    @staticmethod
    def from_dataset(dataset):
        return DatasetLoader(dataset)

    @staticmethod
    def from_filelist(filelist):
        return FileListLoader(filelist)


class GeneratorLoader(DataLoader):
    def __init__(self,
                 feed_list=None,
                 capacity=None,
                 use_double_buffer=True,
                 iterable=True,
                 return_list=False):
        self._tensor_reader = None
        self._places = None
        self._thread = None
        self._feed_list = feed_list
        if not capacity:
            raise ValueError("Please give value to capacity.")
        # force to use iterable mode under dygraph mode
        if in_dygraph_mode():
            if not iterable:
                warnings.warn(
                    "Please NOTE: dygraph can support iterable mode only.")
            self._iterable = True
            if not return_list:
                warnings.warn(
                    "Please NOTE: dygraph can support return as list only.")
            self._return_list = True
        else:
            self._iterable = iterable
            self._return_list = return_list
            if not self._feed_list:
                raise Exception("Feed list must be given under static mode.")
        self._use_double_buffer = use_double_buffer
        self._capacity = capacity
        if not self._iterable:
            self._init_non_iterable()

    def _wait_thread_ends(self):
        # Get self._thread first to prevent data race, because __thread_main__ 
        # would set self._thread be None at the end
        thread = self._thread
        if thread is not None and self._iterable:
            self._queue.close()
            thread.join()

    def _init_iterable(self):
        self._wait_thread_ends()
        if in_dygraph_mode():
            self._var_names = []
        else:
            self._var_names = [v.name for v in self._feed_list]
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

        queue_name = data_loader_unique_name_generator(
            'lod_tensor_blocking_queue')
        reader_name = data_loader_unique_name_generator('create_py_reader')
        double_buffer_name = data_loader_unique_name_generator('double_buffer')

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

    def __iter__(self):
        assert self.iterable, "DataLoader is not iterable"
        assert self._tensor_reader is not None, \
            "Data source of DataLoader has not set yet"

        self._init_iterable()
        self._start()
        return self

    def __next__(self):
        try:
            if not in_dygraph_mode():
                if self._return_list:
                    return self._reader.read_next_list()
                else:
                    return self._reader.read_next()
            else:
                ret = self._reader.read_next_list()[0]
                return [dygraph.base.to_variable(np.array(v)) for v in ret]
        except StopIteration:
            self._queue.close()
            self._reset()
            six.reraise(*sys.exc_info())

    def start(self):
        if not in_dygraph_mode():
            assert not self._iterable, "start() cannot be called when DataLoader is iterable"
            self._start()

    def reset(self):
        if not in_dygraph_mode():
            assert not self._iterable, "reset() cannot be called when DataLoader is iterable"
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
                self._thread = None
            except Exception as ex:
                self._queue.close()
                self._thread = None
                logging.warn('Your reader has raised an exception!')
                six.reraise(*sys.exc_info())

        self._thread = threading.Thread(target=__thread_main__)
        self._thread.daemon = True
        self._thread.start()

    def _reset(self):
        self._reader.reset()
        thread = self._thread
        if thread is not None:
            thread.join()

    def set_sample_generator(self,
                             reader,
                             batch_size,
                             drop_last=True,
                             places=None):
        assert batch_size > 0, "batch_size must be larger than 0"
        if not in_dygraph_mode():
            has_lod = False
            for f in self._feed_list:
                if f.lod_level != 0:
                    has_lod = True
                    break

            if has_lod:
                self.set_sample_list_generator(
                    paddle.batch(
                        reader, batch_size=batch_size, drop_last=drop_last),
                    places=places)
            else:
                reader = BatchedTensorProvider(
                    feed_list=self._feed_list,
                    place=core.CPUPlace(),
                    batch_size=batch_size,
                    generator=reader,
                    drop_last=drop_last)
                self.set_batch_generator(reader, places=places)
        else:
            self.set_sample_list_generator(
                paddle.batch(
                    reader, batch_size=batch_size, drop_last=drop_last),
                places=places)
        return self

    def set_sample_list_generator(self, reader, places=None):
        if not in_dygraph_mode():
            with program_guard(Program(), Program()):
                feeder = DataFeeder(
                    feed_list=self._feed_list, place=core.CPUPlace())
                paddle_reader = feeder.decorate_reader(
                    reader, multi_devices=False)

            def __tensor_reader_impl__():
                for slots in paddle_reader():
                    yield [slots[var.name] for var in self._feed_list]
        else:
            provider = ListTensorProvider(reader, places)

            def __tensor_reader_impl__():
                for slots in provider():
                    yield slots[0]

        self.set_batch_generator(__tensor_reader_impl__, places)
        return self

    def set_batch_generator(self, reader, places=None):
        self._tensor_reader = reader
        if self._iterable:
            assert places is not None, "Places cannot be None when DataLoader is iterable"
            self._places = _convert_places(places)
            if in_dygraph_mode():
                assert len(self._places
                           ) == 1, "Number of places must be 1 in dygraph mode"
        else:
            if places is not None:
                logging.info(
                    'places would be ommited when DataLoader is not iterable')
        return self


class PyReader(DataLoader):
    """
    Create a reader object for data feeding in Python. 
    Data would be prefetched using Python thread and be pushed
    into a queue asynchronously. Data in the queue would be extracted 
    automatically when `Executor.run(...)` is called.

    Args:  
        feed_list (list(Variable)|tuple(Variable)): feed variable list.
            The variables should be created by :code:`fluid.layers.data()`.
            it can be None under iterable mode.
        capacity (int): capacity of the queue maintained in PyReader object. 
        use_double_buffer (bool): whether to use double_buffer_reader to 
            speed up data feeding. 
        iterable (bool): whether the created reader object is iterable.   
        return_list (bool): whether the return value presented as list.
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

           import paddle
           import paddle.fluid as fluid
           import numpy as np

           EPOCH_NUM = 3
           ITER_NUM = 5
           BATCH_SIZE = 3

           def reader_creator_random_image_and_label(height, width):
               def reader():
                   for i in range(ITER_NUM):
                       fake_image = np.random.uniform(low=0,
                                                      high=255,
                                                      size=[height, width])
                       fake_label = np.ones([1])
                       yield fake_image, fake_label
               return reader

           image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
           label = fluid.layers.data(name='label', shape=[1], dtype='int64')

           reader = fluid.io.PyReader(feed_list=[image, label],
                                      capacity=4,
                                      iterable=False)

           user_defined_reader = reader_creator_random_image_and_label(784, 784)
           reader.decorate_sample_list_generator(
               paddle.batch(user_defined_reader, batch_size=BATCH_SIZE))
           # definition of network is omitted
           executor = fluid.Executor(fluid.CUDAPlace(0))
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

           import paddle
           import paddle.fluid as fluid
           import numpy as np

           EPOCH_NUM = 3
           ITER_NUM = 5
           BATCH_SIZE = 10

           def reader_creator_random_image(height, width):
               def reader():
                   for i in range(ITER_NUM):
                       yield np.random.uniform(low=0, high=255, size=[height, width]),
               return reader

           image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
           reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=True, return_list=False)

           user_defined_reader = reader_creator_random_image(784, 784)
           reader.decorate_sample_list_generator(
               paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
               fluid.core.CUDAPlace(0))
           # definition of network is omitted
           executor = fluid.Executor(fluid.CUDAPlace(0))
           executor.run(fluid.default_main_program())

           for _ in range(EPOCH_NUM):
               for data in reader():
                   executor.run(feed=data)


        3. If return_list=True, the return values would be presented as list instead of dict`.

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            EPOCH_NUM = 3
            ITER_NUM = 5
            BATCH_SIZE = 10

            def reader_creator_random_image(height, width):
                def reader():
                    for i in range(ITER_NUM):
                        yield np.random.uniform(low=0, high=255, size=[height, width]),
                return reader

            image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
            reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=True, return_list=True)

            user_defined_reader = reader_creator_random_image(784, 784)
            reader.decorate_sample_list_generator(
                paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
                fluid.core.CPUPlace())
            # definition of network is omitted
            executor = fluid.Executor(fluid.core.CPUPlace())
            executor.run(fluid.default_main_program())

            for _ in range(EPOCH_NUM):
                for data in reader():
                    executor.run(feed={"image": data[0]})
    """

    def __init__(self,
                 feed_list=None,
                 capacity=None,
                 use_double_buffer=True,
                 iterable=True,
                 return_list=False):
        self._loader = DataLoaderFactory.from_generator(
            feed_list, capacity, use_double_buffer, iterable, return_list)

    @property
    def queue(self):
        return self._loader.queue

    @property
    def iterable(self):
        return self._loader.iterable

    def __iter__(self):
        return self._loader.__iter__()

    def __next__(self):
        return self._loader.__next__()

    def _reset(self):
        self._reader.reset()
        thread = self._thread
        if thread is not None:
            thread.join()

    def start(self):
        '''
        Start the data feeding thread. 
        Can only call when the reader object is not iterable.  
        
	    Example:
	        .. code-block:: python
    
                import paddle
                import paddle.fluid as fluid
                import numpy as np

                BATCH_SIZE = 10

                def generator():
                    for i in range(5):
                        yield np.random.uniform(low=0, high=255, size=[784, 784]),

                image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
                reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=False)
                reader.decorate_sample_list_generator(
                    paddle.batch(generator, batch_size=BATCH_SIZE))

                executor = fluid.Executor(fluid.CUDAPlace(0))
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
        self._loader.start()

    def reset(self):
        '''
        Reset the reader object when :code:`fluid.core.EOFException` raises. 
        Can only call when the reader object is not iterable.
        
        Example:
            .. code-block:: python

                import paddle
                import paddle.fluid as fluid
                import numpy as np

                BATCH_SIZE = 10

                def generator():
                    for i in range(5):
                        yield np.random.uniform(low=0, high=255, size=[784, 784]),

                image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
                reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=False)
                reader.decorate_sample_list_generator(
                    paddle.batch(generator, batch_size=BATCH_SIZE))

                executor = fluid.Executor(fluid.CUDAPlace(0))
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
        self._loader.reset()

    def decorate_sample_generator(self,
                                  sample_generator,
                                  batch_size,
                                  drop_last=True,
                                  places=None):
        '''
        Set the data source of the PyReader object.
        
        The provided :code:`sample_generator` should be a Python generator,
        which yields list(numpy.ndarray)-typed data of each sample.

        :code:`places` must be set when the PyReader object is iterable.

        If all inputs have no lods, this method is faster than 
        :code:`decorate_sample_list_generator(paddle.batch(sample_generator, ...))` .

        Args:
            sample_generator (generator): Python generator that yields
                list(numpy.ndarray)-typed sample data.
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
                ITER_NUM = 15
                BATCH_SIZE = 3

                def random_image_and_label_generator(height, width):
                    def generator():
                        for i in range(ITER_NUM):
                            fake_image = np.random.uniform(low=0,
                                                           high=255,
                                                           size=[height, width])
                            fake_label = np.array([1])
                            yield fake_image, fake_label
                    return generator

                image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
                label = fluid.layers.data(name='label', shape=[1], dtype='int32')
                reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)

                user_defined_generator = random_image_and_label_generator(784, 784)
                reader.decorate_sample_generator(user_defined_generator,
                                                 batch_size=BATCH_SIZE,
                                                 places=[fluid.CUDAPlace(0)])
                # definition of network is omitted
                executor = fluid.Executor(fluid.CUDAPlace(0))
                executor.run(fluid.default_main_program())

                for _ in range(EPOCH_NUM):
                    for data in reader():
                        executor.run(feed=data)
    
        '''
        self._loader.set_sample_generator(sample_generator, batch_size,
                                          drop_last, places)

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

                import paddle
                import paddle.fluid as fluid
                import numpy as np

                EPOCH_NUM = 3
                ITER_NUM = 15
                BATCH_SIZE = 3

                def random_image_and_label_generator(height, width):
                    def generator():
                        for i in range(ITER_NUM):
                            fake_image = np.random.uniform(low=0,
                                                           high=255,
                                                           size=[height, width])
                            fake_label = np.ones([1])
                            yield fake_image, fake_label
                    return generator

                image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
                label = fluid.layers.data(name='label', shape=[1], dtype='int32')
                reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)

                user_defined_generator = random_image_and_label_generator(784, 784)
                reader.decorate_sample_list_generator(
                    paddle.batch(user_defined_generator, batch_size=BATCH_SIZE),
                    fluid.core.CUDAPlace(0))
                # definition of network is omitted
                executor = fluid.Executor(fluid.core.CUDAPlace(0))
                executor.run(fluid.default_main_program())

                for _ in range(EPOCH_NUM):
                    for data in reader():
                        executor.run(feed=data)
                 
        '''
        self._loader.set_sample_list_generator(reader, places)

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
                ITER_NUM = 15
                BATCH_SIZE = 3

                def random_image_and_label_generator(height, width):
                    def generator():
                        for i in range(ITER_NUM):
                            batch_image = np.random.uniform(low=0,
                                                            high=255,
                                                            size=[BATCH_SIZE, height, width])
                            batch_label = np.ones([BATCH_SIZE, 1])
                            yield batch_image, batch_label
                    return generator

                image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
                label = fluid.layers.data(name='label', shape=[1], dtype='int32')
                reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)

                user_defined_generator = random_image_and_label_generator(784, 784)
                reader.decorate_batch_generator(user_defined_generator, fluid.CUDAPlace(0))
                # definition of network is omitted
                executor = fluid.Executor(fluid.CUDAPlace(0))
                executor.run(fluid.default_main_program())

                for _ in range(EPOCH_NUM):
                    for data in reader():
                        executor.run(feed=data)

        '''
        self._loader.set_batch_generator(reader, places)


class DatasetLoader(DataLoader):
    def __init__(self, dataset):
        self._dataset = dataset


class FileListLoader(DataLoader):
    def __init__(self, filelist):
        self._filelist = filelist
