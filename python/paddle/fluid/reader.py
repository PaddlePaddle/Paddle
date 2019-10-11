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
from .framework import Program, Variable, program_guard, default_main_program, default_startup_program, in_dygraph_mode, cpu_places
from .executor import global_scope
from .data_feeder import DataFeeder, BatchedTensorProvider, ListTensorProvider
from .layers.io import monkey_patch_reader_methods, _copy_reader_var_, double_buffer
from .unique_name import UniqueNameGenerator
import logging
from .dataset import DatasetBase, InMemoryDataset

__all__ = ['PyReader', 'DataLoader']

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


class DataLoaderBase(object):
    def __init__(self):
        self._places = None

    def __call__(self):
        return self

    def next(self):
        '''
        Get the next item in the DataLoader object. This method    
        should not be called by users directly. It is used for
        implementing iterator protocol of Python 2.x inside
        PaddlePaddle framework.
        '''
        return self.__next__()

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()


class DataLoader(object):
    @staticmethod
    def from_generator(feed_list=None,
                       capacity=None,
                       use_double_buffer=True,
                       iterable=True,
                       return_list=False):
        """
        Create a DataLoader object for loading data from Python generator. 
        Data would be prefetched using Python thread and be pushed
        into a queue asynchronously.

        The created DataLoader object provides 3 methods to set the data source
        :code:`set_sample_generator` , :code:`set_sample_list_generator` and 
        :code:`set_batch_generator` . Please see the following example codes
        to know their usages.

        If iterable = True, the created DataLoader object is a Python generator
        object, which is iterable using for-range loop.

        If iterable = False, the created DataLoader object provides 
        :code:`start()` and :code:`reset()` method to control the data reading
        process. This mode is designed to be compatible with the 
        :code:`fluid.layers.py_reader` interface. Users can migrate the codes   
        from :code:`fluid.layers.py_reader` to :code:`fluid.io.DataLoader` 
        easily when using iterable=False. 

        Args:  
            feed_list (list(Variable)|tuple(Variable)): feed variable list.
                The variables should be created by :code:`fluid.data()`.
            capacity (int): capacity of the queue maintained in DataLoader.
                The unit is batch number. Set larger capacity if your reader 
                is fast. 
            use_double_buffer (bool): whether to use double_buffer_reader. 
                If use_double_buffer=True, the DataLoader would prefetch next 
                batch data asynchronously, so it would speed up data feeding 
                and occupies a little more CPU or GPU memory, i.e., the memory
                of one batch input data. 
            iterable (bool): whether the created DataLoader is iterable. 
            return_list (bool): whether the return value on each device is 
                presented as a list. It is only valid when iterable=True. 
                If return_list=False, the return value on each device would 
                be a dict of str -> LoDTensor, where the key of the dict is 
                the name of each feeded variables. If return_list=True, the 
                return value on each device would be a list(LoDTensor). It is
                recommended to use return_list=False in static graph mode and
                use return_list=True in dygraph mode.   

        Returns:
            loader (DataLoader): the created DataLoader object.

        Examples:
            
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                BATCH_NUM = 10 
                BATCH_SIZE = 16
                EPOCH_NUM = 4

                CLASS_NUM = 10

                ITERABLE = True # whether the created DataLoader object is iterable
                USE_GPU = False # whether to use GPU

                DATA_FORMAT = 'batch_generator' # data format of data source user provides 

                def simple_net(image, label):
                    fc_tmp = fluid.layers.fc(image, size=CLASS_NUM)
                    cross_entropy = fluid.layers.softmax_with_cross_entropy(image, label)
                    loss = fluid.layers.reduce_mean(cross_entropy)
                    sgd = fluid.optimizer.SGD(learning_rate=1e-3)
                    sgd.minimize(loss)
                    return loss

                def get_random_images_and_labels(image_shape, label_shape):
                    image = np.random.random(size=image_shape).astype('float32')
                    label = np.random.random(size=label_shape).astype('int64')
                    return image, label

                # If the data generator yields one sample each time,
                # use DataLoader.set_sample_generator to set the data source.
                def sample_generator_creator(): 
                    def __reader__():
                        for _ in range(BATCH_NUM * BATCH_SIZE):
                            image, label = get_random_images_and_labels([784], [1])
                            yield image, label

                    return __reader__

                # If the data generator yield list of samples each time,
                # use DataLoader.set_sample_list_generator to set the data source.
                def sample_list_generator_creator():
                    def __reader__():
                        for _ in range(BATCH_NUM): 
                            sample_list = []
                            for _ in range(BATCH_SIZE):
                                image, label = get_random_images_and_labels([784], [1])
                                sample_list.append([image, label])

                            yield sample_list

                    return __reader__ 

                # If the data generator yields a batch each time, 
                # use DataLoader.set_batch_generator to set the data source.
                def batch_generator_creator():
                    def __reader__():
                        for _ in range(BATCH_NUM):
                            batch_image, batch_label = get_random_images_and_labels([BATCH_SIZE, 784], [BATCH_SIZE, 1]) 
                            yield batch_image, batch_label

                    return __reader__

                # If DataLoader is iterable, use for loop to train the network 
                def train_iterable(exe, prog, loss, loader):
                    for _ in range(EPOCH_NUM):
                        for data in loader():
                            exe.run(prog, feed=data, fetch_list=[loss])

                # If DataLoader is not iterable, use start() and reset() method to control the process 
                def train_non_iterable(exe, prog, loss, loader):
                    for _ in range(EPOCH_NUM):
                        loader.start() # call DataLoader.start() before each epoch starts
                        try:
                            while True:
                                exe.run(prog, fetch_list=[loss])
                        except fluid.core.EOFException:
                            loader.reset() # call DataLoader.reset() after catching EOFException 

                def set_data_source(loader, places):
                    if DATA_FORMAT == 'sample_generator':
                        loader.set_sample_generator(sample_generator_creator(), batch_size=BATCH_SIZE, drop_last=True, places=places)
                    elif DATA_FORMAT == 'sample_list_generator':
                        loader.set_sample_list_generator(sample_list_generator_creator(), places=places)
                    elif DATA_FORMAT == 'batch_generator':
                        loader.set_batch_generator(batch_generator_creator(), places=places)
                    else:
                        raise ValueError('Unsupported data format')

                image = fluid.data(name='image', shape=[None, 784], dtype='float32')
                label = fluid.data(name='label', shape=[None, 1], dtype='int64')

                # Define DataLoader 
                loader = fluid.io.DataLoader.from_generator(feed_list=[image, label], capacity=16, iterable=ITERABLE)

                # Define network
                loss = simple_net(image, label)

                # Set data source of DataLoader
                #
                # If DataLoader is iterable, places must be given and the number of places must be the same with device number.  
                #  - If you are using GPU, call `fluid.cuda_places()` to get all GPU places. 
                #  - If you are using CPU, call `fluid.cpu_places()` to get all CPU places. 
                # 
                # If DataLoader is not iterable, places can be None.
                places = fluid.cuda_places() if USE_GPU else fluid.cpu_places()
                set_data_source(loader, places)

                exe = fluid.Executor(places[0])
                exe.run(fluid.default_startup_program())

                prog = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(loss_name=loss.name)

                if loader.iterable:
                    train_iterable(exe, prog, loss, loader)
                else:
                    train_non_iterable(exe, prog, loss, loader)


                '''
                Users can use return_list = True in dygraph mode. 
                '''
                with fluid.dygraph.guard(places[0]):
                    loader = fluid.io.DataLoader.from_generator(capacity=2, return_list=True)
                    set_data_source(loader, places[0]) 
                    for image, label in loader():
                        relu = fluid.layers.relu(image)
                        assert image.shape == [BATCH_SIZE, 784] 
                        assert label.shape == [BATCH_SIZE, 1]
                        assert relu.shape == [BATCH_SIZE, 784]
        """
        return GeneratorLoader(feed_list, capacity, use_double_buffer, iterable,
                               return_list)

    @staticmethod
    def from_dataset(dataset, places, drop_last=True):
        """
        Create an iterable DataLoader object for loading data from Dataset.    
        Dataset is only supported in Linux system currently.

        Args:
            dataset (InMemoryDataset|QueueDataset): the dataset object.
            places (list(CUDAPlace)|list(CPUPlace)): places where the result 
                data should be converted.   
            drop_last (bool): whether to drop the last batch whose sample 
                number is less than batch size. If drop_last = True, they
                would be dropped. If drop_last = False, they would be kept. 

        Returns:
            loader (DataLoader): the created DataLoader object, which can be 
                treated as a Python generator.   

        Examples:

            .. code-block:: python

                import paddle.fluid as fluid

                image = fluid.data(name='image', shape=[None, 784], dtype='float32')
                label = fluid.data(name='label', shape=[None, 1], dtype='int64')

                dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
                dataset.set_batch_size(32)
                dataset.set_filelist(['a.txt', 'b.txt', 'c.txt'])
                dataset.set_use_var([image, label])
                dataset.set_pipe_command('cat') 

                loader = fluid.io.DataLoader.from_dataset(dataset, fluid.cpu_places())
        """
        return DatasetLoader(dataset, places, drop_last)


class GeneratorLoader(DataLoaderBase):
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
                    "Please NOTE: dygraph can support iterable mode only. Change to iterable mode."
                )
            self._iterable = True
            if not return_list:
                warnings.warn(
                    "Please NOTE: dygraph can support return as list only. Change to return as list."
                )
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

    @classmethod
    def _check_input_array(cls, item):
        arr = np.array(item)
        if arr.dtype == np.object:
            raise TypeError((
                "\n\tFaild to convert input data to a regular ndarray :\n\t* Usually "
                "this means the input data contains nested lists with different lengths. "
                "\n\t* Check the reader function passed to 'decorate_batch_generator'"
                " to locate the data causes this issue.\n\t* Please consider using "
                "'fluid.create_lod_tensor' to convert it to a LoD-Tensor."))

    def _start(self):
        def __thread_main__():
            try:
                for tensors in self._tensor_reader():
                    array = core.LoDTensorArray()
                    for item in tensors:
                        if not isinstance(item, core.LoDTensor):
                            self._check_input_array(item)
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


class PyReader(DataLoaderBase):
    """
    Create a reader object for data feeding in Python. 
    Data would be prefetched using Python thread and be pushed
    into a queue asynchronously. Data in the queue would be extracted 
    automatically when `Executor.run(...)` is called.

    Args:  
        feed_list (list(Variable)|tuple(Variable)): feed variable list.
            The variables should be created by :code:`fluid.layers.data()`.
        capacity (int): capacity of the queue maintained in PyReader.
            The unit is batch number. Set larger capacity if your reader 
            is fast. 
        use_double_buffer (bool): whether to use double_buffer_reader. 
            If use_double_buffer=True, PyReader would prefetch next 
            batch data asynchronously, so it would speed up data feeding 
            and occupies a little more CPU or GPU memory, i.e., the memory
            of one batch input data. 
        iterable (bool): whether the created PyReader is iterable. 
        return_list (bool): whether the return value on each device is 
            presented as a list. It is only valid when iterable=True. 
            If return_list=False, the return value on each device would 
            be a dict of str -> LoDTensor, where the key of the dict is 
            the name of each feeded variables. If return_list=True, the 
            return value on each device would be a list(LoDTensor). It is
            recommended to use return_list=False in static graph mode and
            use return_list=True in dygraph mode. 

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


        3. If return_list=True, the return values would be presented as list instead of dict. 
           This is usually used in dygraph mode.

        .. code-block:: python

           import paddle
           import paddle.fluid as fluid
           import numpy as np

           ITER_NUM = 5
           BATCH_SIZE = 10

           def reader_creator_random_image(height, width):
               def reader():
                   for i in range(ITER_NUM):
                       yield np.random.uniform(low=0, high=255, size=[height, width]), \
                           np.random.random_integers(low=0, high=9, size=[1])
               return reader

           place = fluid.CPUPlace()
           with fluid.dygraph.guard(place):
               py_reader = fluid.io.PyReader(capacity=2, return_list=True)
               user_defined_reader = reader_creator_random_image(784, 784)
               py_reader.decorate_sample_list_generator(
                   paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
                   place)
               for image, label in py_reader():
                   relu = fluid.layers.relu(image)
    """

    def __init__(self,
                 feed_list=None,
                 capacity=None,
                 use_double_buffer=True,
                 iterable=True,
                 return_list=False):
        self._loader = DataLoader.from_generator(
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


class DatasetLoader(DataLoaderBase):
    def __init__(self, dataset, places, drop_last):
        assert isinstance(dataset,
                          DatasetBase), "dataset must be type of DatasetBase"
        assert not in_dygraph_mode(
        ), "DatasetLoader is not supported in dygraph mode yet"

        thread_num = len(places)

        assert len(dataset.filelist) >= thread_num, \
            "Filelist number of dataset {} must be not less than place number {}".format(len(dataset.filelist), thread_num)

        if dataset.thread_num != 0 and dataset.thread_num != thread_num:
            logging.warn('thread_num {} which is set in Dataset is ignored'.
                         format(dataset.thread_num))

        dataset.set_thread(thread_num)

        if isinstance(dataset,
                      InMemoryDataset) and dataset.queue_num > thread_num:
            logging.warn("queue_num {} which is set in Dataset is ignored".
                         format(dataset.queue_num))
            dataset.set_queue_num(thread_num)

        self._dataset = dataset
        use_slots = [
            slot.name for slot in dataset.proto_desc.multi_slot_desc.slots
            if slot.is_used
        ]

        self._iterable_dataset = core.IterableDatasetWrapper(
            dataset.dataset, use_slots,
            _convert_places(places), dataset.proto_desc.batch_size, drop_last)

    def __iter__(self):
        self._dataset._finish_to_run()
        self._dataset._prepare_to_run()
        self._iterable_dataset._start()
        return self

    def __next__(self):
        return self._iterable_dataset._next()
