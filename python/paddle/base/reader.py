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

import logging
import multiprocessing
import queue
import sys
import threading
import warnings

import numpy as np

import paddle
from paddle.base.framework import _set_expected_place
from paddle.pir.core import datatype_to_vartype

from . import core
from .data_feeder import BatchedTensorProvider, DataFeeder
from .executor import global_scope
from .framework import (
    Program,
    _current_expected_place,
    _get_paddle_place,
    _get_paddle_place_list,
    default_main_program,
    default_startup_program,
    in_dygraph_mode,
    in_pir_mode,
    program_guard,
)
from .layers.io import (
    __create_unshared_decorated_reader__,
    _copy_reader_var_,
    monkey_patch_reader_methods,
)
from .multiprocess_utils import (  # noqa: F401
    CleanupFuncRegistrar,
    _cleanup,
    _cleanup_mmap,
    _set_SIGCHLD_handler,
    multiprocess_queue_set,
)
from .unique_name import UniqueNameGenerator

# NOTE: [ avoid hanging & failed quickly ] These value is used in getting data from another process
QUEUE_GET_TIMEOUT = 60

__all__ = []

data_loader_unique_name_generator = UniqueNameGenerator()

KEEP_DATA_LOADER_ORDER = True
USE_PINNED_MEMORY = None


def keep_data_loader_order(*args):
    global KEEP_DATA_LOADER_ORDER
    if len(args) == 0:
        return KEEP_DATA_LOADER_ORDER
    else:
        assert len(args) == 1 and isinstance(args[0], bool)
        KEEP_DATA_LOADER_ORDER = args[0]


def use_pinned_memory(*args):
    global USE_PINNED_MEMORY
    if len(args) == 0:
        return USE_PINNED_MEMORY
    else:
        assert len(args) == 1 and isinstance(args[0], bool)
        USE_PINNED_MEMORY = args[0]


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


# NOTE(chenweihang): _reader_process_loop must be top level method to be pickled
def _reader_process_loop(
    batch_reader, data_queue, dataloader_use_file_descriptor=True
):
    try:
        # set signal handler
        core._set_process_signal_handler()
        if not dataloader_use_file_descriptor:
            # set dataloader_use_file_descriptor to false to avoid use descriptor.
            paddle.base.core.globals()[
                "FLAGS_dataloader_use_file_descriptor"
            ] = False

        # NOTE: [ mmap files clear ] When the child process exits unexpectedly,
        # some shared memory objects may have been applied for but have not yet
        # been put into the inter-process Queue. This part of the object needs
        # to be cleaned up when the process ends.
        CleanupFuncRegistrar.register(_cleanup_mmap)

        for batch in batch_reader():
            tensor_list = core._convert_to_tensor_list(batch)
            data_queue.put(tensor_list)
            core._remove_tensor_list_mmap_fds(tensor_list)
        data_queue.put(None)
    except KeyboardInterrupt:
        # NOTE: Main process will raise KeyboardInterrupt anyways, ignore it in child process
        pass
    except:
        raise


class DataLoaderBase:
    def __init__(self):
        self._places = None

    def __call__(self):
        return self

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()

    @classmethod
    def _check_input_array(cls, item):
        arr = np.asarray(item)
        if arr.dtype == np.object_:
            raise TypeError(
                "\n\tFailed to convert input data to a regular ndarray :\n\t* Usually "
                "this means the input data contains nested lists with different lengths. "
                "\n\t* Check the reader function passed to 'decorate_batch_generator'"
                " to locate the data causes this issue.\n\t* Please consider using "
                "'base.create_lod_tensor' to convert it to a LoD-Tensor."
            )
        return arr


class DataLoader:
    @staticmethod
    def from_generator(
        feed_list=None,
        capacity=None,
        use_double_buffer=True,
        iterable=True,
        return_list=False,
        use_multiprocess=False,
        drop_last=True,
    ):
        """
        .. warning::
          This API will be deprecated in the future, it is recommended to use
          :code:`paddle.io.DataLoader` which supports multi-processes acceleration.

        .. note::
          **The framework ensures that the data loading order of DataLoader is exactly the same as the user-defined data source.**

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
        process.

        Args:
            feed_list (list(Tensor)|tuple(Tensor)): feed Tensor list.
                The Tensors should be created by :code:`paddle.static.data()`.
            capacity (int): capacity of the queue maintained in DataLoader.
                The unit is batch number. Set larger capacity if your reader
                is fast.
            use_double_buffer (bool, optional): whether to use double_buffer_reader.
                If use_double_buffer=True, the DataLoader would prefetch next
                batch data asynchronously, so it would speed up data feeding
                and occupies a little more CPU or GPU memory, i.e., the memory
                of one batch input data.
            iterable (bool, optional): whether the created DataLoader is iterable.
            return_list (bool, optional): whether the return value on each device is
                presented as a list. It is only valid when iterable=True.
                If return_list=False, the return value on each device would
                be a dict of str -> LoDTensor, where the key of the dict is
                the name of each fed Tensors. If return_list=True, the
                return value on each device would be a list(LoDTensor). It is
                recommended to use return_list=False in static graph mode and
                use return_list=True in dygraph mode.
            use_multiprocess (bool, optional): whether to use multi-process to
                speed up the data loading process in dygraph. Note: this parameter
                only can be used in the dygraph mode. In the static graph mode,
                whether this parameter is set or not has no effect.
                The Default value is False.
            drop_last (bool, optional): whether to drop the last batches whose
                number is less than the CPU core/GPU card number. The default
                value is True. In training phase, users should not set drop_last=False,
                because all CPU cores/GPU cards must read data from DataLoader.
                In inference phase, users can set drop_last=False, so that the
                last batches whose number is less than the CPU core/GPU card
                number can be tested.

        Returns:
            loader (DataLoader): the created DataLoader object.

        Examples:
            .. code-block:: python
                :name: example_1

                >>> # Example in static graph mode

                >>> import numpy as np

                >>> import paddle
                >>> import paddle.static as static
                >>> import paddle.nn.functional as F


                >>> BATCH_NUM = 10
                >>> BATCH_SIZE = 16
                >>> EPOCH_NUM = 4

                >>> CLASS_NUM = 10

                >>> ITERABLE = True # whether the created DataLoader object is iterable
                >>> USE_GPU = False # whether to use GPU

                >>> DATA_FORMAT = 'batch_generator' # data format of data source user provides

                >>> paddle.enable_static()

                >>> def simple_net(image, label):
                ...     fc_tmp = static.nn.fc(image, size=CLASS_NUM)
                ...     cross_entropy = F.softmax_with_cross_entropy(image, label)
                ...     loss = paddle.mean(cross_entropy)
                ...     sgd = paddle.optimizer.SGD(learning_rate=1e-3)
                ...     sgd.minimize(loss)
                ...     return loss
                ...
                >>> def get_random_images_and_labels(image_shape, label_shape):
                ...     image = np.random.random(size=image_shape).astype('float32')
                ...     label = np.random.random(size=label_shape).astype('int64')
                ...     return image, label
                ...
                >>> # If the data generator yields one sample each time,
                >>> # use DataLoader.set_sample_generator to set the data source.
                >>> def sample_generator_creator():
                ...     def __reader__():
                ...         for _ in range(BATCH_NUM * BATCH_SIZE):
                ...             image, label = get_random_images_and_labels([784], [1])
                ...             yield image, label
                ...
                ...     return __reader__
                ...
                >>> # If the data generator yield list of samples each time,
                >>> # use DataLoader.set_sample_list_generator to set the data source.
                >>> def sample_list_generator_creator():
                ...     def __reader__():
                ...         for _ in range(BATCH_NUM):
                ...             sample_list = []
                ...             for _ in range(BATCH_SIZE):
                ...                 image, label = get_random_images_and_labels([784], [1])
                ...                 sample_list.append([image, label])
                ...
                ...             yield sample_list
                ...
                ...     return __reader__
                ...
                >>> # If the data generator yields a batch each time,
                >>> # use DataLoader.set_batch_generator to set the data source.
                >>> def batch_generator_creator():
                ...     def __reader__():
                ...         for _ in range(BATCH_NUM):
                ...             batch_image, batch_label = get_random_images_and_labels([BATCH_SIZE, 784], [BATCH_SIZE, 1])
                ...             yield batch_image, batch_label
                ...
                ...     return __reader__
                ...
                >>> # If DataLoader is iterable, use for loop to train the network
                >>> def train_iterable(exe, prog, loss, loader):
                ...     for _ in range(EPOCH_NUM):
                ...         for data in loader():
                ...             exe.run(prog, feed=data, fetch_list=[loss])
                ...
                >>> # If DataLoader is not iterable, use start() and reset() method to control the process
                >>> def train_non_iterable(exe, prog, loss, loader):
                ...     for _ in range(EPOCH_NUM):
                ...         loader.start() # call DataLoader.start() before each epoch starts
                ...         try:
                ...             while True:
                ...                 exe.run(prog, fetch_list=[loss])
                ...         except paddle.core.EOFException:
                ...             loader.reset() # call DataLoader.reset() after catching EOFException
                ...
                >>> def set_data_source(loader, places):
                ...     if DATA_FORMAT == 'sample_generator':
                ...         loader.set_sample_generator(sample_generator_creator(), batch_size=BATCH_SIZE, drop_last=True, places=places)
                ...     elif DATA_FORMAT == 'sample_list_generator':
                ...         loader.set_sample_list_generator(sample_list_generator_creator(), places=places)
                ...     elif DATA_FORMAT == 'batch_generator':
                ...         loader.set_batch_generator(batch_generator_creator(), places=places)
                ...     else:
                ...         raise ValueError('Unsupported data format')
                ...
                >>> image = static.data(name='image', shape=[None, 784], dtype='float32')
                >>> label = static.data(name='label', shape=[None, 1], dtype='int64')

                >>> # Define DataLoader
                >>> loader = paddle.base.io.DataLoader.from_generator(feed_list=[image, label], capacity=16, iterable=ITERABLE)

                >>> # Define network
                >>> loss = simple_net(image, label)

                >>> places = static.cuda_places() if USE_GPU else static.cpu_places()
                >>> set_data_source(loader, places)

                >>> exe = static.Executor(places[0])
                >>> exe.run(static.default_startup_program())

                >>> prog = static.CompiledProgram(static.default_main_program())
                >>> if loader.iterable:
                ...     train_iterable(exe, prog, loss, loader)
                >>> else:
                ...     train_non_iterable(exe, prog, loss, loader)

            .. code-block:: python
                :name: example_2

                >>> # Example in dynamic graph mode.

                >>> import numpy as np

                >>> import paddle
                >>> import paddle.nn as nn
                >>> import paddle.optimizer as opt
                >>> import paddle.distributed as dist

                >>> BATCH_SIZE = 16
                >>> BATCH_NUM = 4
                >>> EPOCH_NUM = 4

                >>> IMAGE_SIZE = 784
                >>> CLASS_NUM = 10

                >>> USE_GPU = False # whether to use GPU

                >>> def _get_random_images_and_labels(image_shape):
                ...         image = np.random.random(size=image_shape).astype('float32')
                ...         label = np.random.randint(0, CLASS_NUM, size=BATCH_SIZE).astype('int64')
                ...         return image, label
                ...
                >>> def __reader__():
                ...         for _ in range(BATCH_NUM):
                ...             batch_image, batch_label = _get_random_images_and_labels(
                ...                 [BATCH_SIZE, IMAGE_SIZE])
                ...             yield batch_image, batch_label
                ...
                >>> def random_batch_reader():
                ...     return __reader__
                ...
                >>> class LinearNet(nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
                ...
                ...     @paddle.jit.to_static
                ...     def forward(self, x):
                ...         return self._linear(x)
                ...
                >>> # set device
                >>> paddle.set_device('gpu' if USE_GPU else 'cpu')

                >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')
                >>> # create network
                >>> layer = LinearNet()
                >>> dp_layer = paddle.DataParallel(layer)
                >>> loss_fn = nn.CrossEntropyLoss()
                >>> adam = opt.Adam(learning_rate=0.001, parameters=dp_layer.parameters())

                >>> # create data loader
                >>> loader = paddle.base.io.DataLoader.from_generator(capacity=5)
                >>> loader.set_batch_generator(random_batch_reader())

                >>> for epoch_id in range(EPOCH_NUM):
                ...     for batch_id, (image, label) in enumerate(loader()):
                ...         out = layer(image)
                ...         loss = loss_fn(out, label)
                ...
                ...         loss.backward()
                ...
                ...         adam.step()
                ...         adam.clear_grad()
                ...         print("Epoch {} batch {}: loss = {}".format(
                ...             epoch_id, batch_id, np.mean(loss.numpy())))
                ...
                >>> # doctest: -SKIP
        """
        if in_dygraph_mode():
            return DygraphGeneratorLoader(
                feed_list,
                capacity,
                use_double_buffer,
                iterable,
                return_list,
                use_multiprocess,
            )
        else:
            return GeneratorLoader(
                feed_list,
                capacity,
                use_double_buffer,
                iterable,
                return_list,
                drop_last,
            )

    @staticmethod
    def from_dataset(dataset, places, drop_last=True):
        """
        .. warning::
          This API will be deprecated in the future, it is recommended to use
          :code:`paddle.io.DataLoader` which supports multi-processes acceleration.

        Create an iterable DataLoader object for loading data from Dataset.
        Dataset is only supported in Linux system currently.

        Args:
            dataset (InMemoryDataset|QueueDataset): the dataset object.
            places (list(CUDAPlace)|list(CPUPlace)|list(str)): places where the result
                data should be converted. If places is list of string, the string in the list
                can be ``cpu``, ``gpu:x`` and ``gpu_pinned``, where x is the index of the GPUs.
            drop_last (bool, optional): whether to drop the last batch whose
                sample number is less than batch size. If drop_last = True,
                they would be dropped. If drop_last = False, they would be kept.

        Returns:
            loader (DataLoader): the created DataLoader object, which can be
                treated as a Python generator.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> import paddle.static as static

                >>> paddle.enable_static()

                >>> image = static.data(name='image', shape=[None, 784], dtype='float32')
                >>> label = static.data(name='label', shape=[None, 1], dtype='int64')

                >>> dataset = paddle.distributed.QueueDataset()
                >>> dataset.init(
                ...     batch_size=32,
                ...     pipe_command='cat',
                ...     use_var=[image, label])
                >>> dataset.set_filelist(['a.txt', 'b.txt', 'c.txt'])

                >>> loader = paddle.base.io.DataLoader.from_dataset(dataset, static.cpu_places())
        """
        return DatasetLoader(dataset, places, drop_last)


class DygraphGeneratorLoader(DataLoaderBase):
    """
    The GeneratorLoader of dygraph

    The multiprocess dygraph GeneratorLoader's most functions are different from
    static graph GeneratorLoader, Separate implementation to keep code readable.
    """

    def __init__(
        self,
        feed_list=None,
        capacity=None,
        use_double_buffer=True,
        iterable=True,
        return_list=True,
        use_multiprocess=False,
    ):
        self._batch_reader = None
        self._places = None
        self._feed_list = feed_list
        self._timeout = QUEUE_GET_TIMEOUT

        if not capacity:
            raise ValueError("Please give value to capacity.")
        self._capacity = capacity
        self._use_double_buffer = use_double_buffer

        if not iterable:
            warnings.warn(
                "Please NOTE: DygraphGeneratorLoader supports iterable mode only. Change to iterable mode."
            )
        self._iterable = True
        if not return_list:
            warnings.warn(
                "Please NOTE: DygraphGeneratorLoader supports returning as list only. Change to return as list."
            )
        self._return_list = True

        # NOTE: the multiprocessing in different platform is incompatible, we will solve it later
        self._use_multiprocess = use_multiprocess
        if self._use_multiprocess and (
            sys.platform == 'darwin' or sys.platform == 'win32'
        ):
            warnings.warn(
                "NOTE: DygraphGeneratorLoader with multiprocess mode is not currently supported on MacOs and Windows."
            )
            self._use_multiprocess = False

        if self._use_multiprocess:
            # NOTE: the multiprocessing.Queue used to save loading data in self._process
            self._data_queue = None
            # NOTE: this process is used to load data asynchronously from self._batch_reader
            self._process = None

        # NOTE: the C++ LoDTensorBlockingQueue instance
        self._blocking_queue = None
        # NOTE: 1. In multiprocess mode, this thread is used to get next batch data from
        # self._data_queue, then push it into self._blocking_queue; 2. In single process
        # mode, this thread is used to get next batch data from self._batch_reader, then
        # push it into self._blocking_queue
        self._thread = None
        self._pin_memory = (
            True if use_pinned_memory() is None else use_pinned_memory()
        )

    @property
    def queue(self):
        return self._blocking_queue

    @property
    def iterable(self):
        return self._iterable

    def _clear_and_remove_data_queue(self):
        if self._data_queue is not None:
            while True:
                try:
                    self._data_queue.get_nowait()
                except queue.Empty:
                    break
            global multiprocess_queue_set
            multiprocess_queue_set.remove(self._data_queue)

    def _wait_thread_ends(self):
        thread = self._thread
        if thread is not None:
            self._blocking_queue.close()
            thread.join()

    def _wait_process_ends(self):
        process = self._process
        if process is not None:
            process.join()
            # erase process id
            core._erase_process_pids(id(self))

    def _init_iterable(self):
        self._wait_thread_ends()
        if self._use_multiprocess:
            self._wait_process_ends()
        self._var_names = []
        self._shapes = []
        self._dtypes = []
        self._need_check_feed = []
        self._blocking_queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._capacity, False
        )
        self._reader = None
        self._reader = core.create_py_reader(
            self.queue,
            self._var_names,
            self._shapes,
            self._dtypes,
            self._need_check_feed,
            self._places,
            self._use_double_buffer,
            True,
            self._pin_memory,
        )

    def _start(self):
        if self._use_multiprocess:
            # clear old _data_queue and remove it from multiprocess_queue_set
            self._clear_and_remove_data_queue()
            # set data_queue and process
            self._data_queue = multiprocessing.Queue(self._capacity)
            # add _data_queue into global queue set
            global multiprocess_queue_set
            multiprocess_queue_set.add(self._data_queue)
            self._process = multiprocessing.Process(
                target=_reader_process_loop,
                args=(self._batch_reader, self._data_queue, False),
            )
            self._process.daemon = True
            self._process.start()

            # Set child process signal handler
            # NOTE: [ avoiding hang ] 1. if the child process dies due to bus error/segfault
            # or just hang, the main process will hang waiting for data, so here need to deal
            # with SIGSEGV and SIGBUS of child process; 2. if the main process end before child
            # process, it shuts the all its daemonic children down with a SIGTERM (instead of
            # joining them without a timeout), so here need to deal with SIGTERM.
            core._set_process_pids(id(self), [self._process.pid])
            _set_SIGCHLD_handler()

            # Set reader_thread
            self._thread_done_event = threading.Event()
            self._thread = threading.Thread(
                target=self._reader_thread_loop_for_multiprocess,
                args=(_current_expected_place(),),
            )
            self._thread.daemon = True
            self._thread.start()
        else:
            self._thread = threading.Thread(
                target=self._reader_thread_loop_for_singleprocess,
                args=(_current_expected_place(),),
            )
            self._thread.daemon = True
            self._thread.start()

    def _reset(self):
        self._reader.reset()
        self._wait_thread_ends()
        if self._use_multiprocess:
            self._wait_process_ends()

    def __iter__(self):
        assert self.iterable, "DataLoader is not iterable"
        assert (
            self._batch_reader is not None
        ), "Data source of DataLoader has not set yet"

        self._init_iterable()
        self._start()
        return self

    def __next__(self):
        try:
            return core.eager.read_next_tensor_list(
                self._reader.read_next_list()[0]
            )
        except StopIteration:
            self._reset()
            raise

    def _exit_thread_expectedly(self):
        self._thread_done_event.set()
        self._blocking_queue.close()

    def _exit_thread_unexpectedly(self):
        self._thread_done_event.set()
        self._blocking_queue.kill()
        logging.error("DataLoader reader thread raised an exception!")

    def _reader_thread_loop_for_multiprocess(self, legacy_expected_place):
        # See _DataLoaderIterSingleProcess._thread_loop() for why set expected place here.
        core.set_current_thread_name("Dataloader_" + str(id(self)))
        _set_expected_place(legacy_expected_place)

        while not self._thread_done_event.is_set():
            try:
                # NOTE: [ avoid hanging ] Even with carefully designed data dependencies
                # (i.e., a put() always corresponding to a get()), hanging on get() can
                # still happen when data in queue is corrupted (e.g., due to
                # Queue.cancel_join_thread or unexpected exit). So we set a timeout whenever
                # we try to get data from `data_queue`
                # NOTE: [ avoid failed quickly ] Here, the time setting of QUEUE_GET_TIMEOUT
                # is relatively long, currently it is 60 seconds, because in some models,
                # if the reader child process starts with a heavy burden, the child process
                # has no enough time to put the data in the queue when the main process
                # start trying to get data from queue. At this time, the child thread needs
                # to wait slightly longer
                tensor_list = self._data_queue.get(timeout=self._timeout)
            except Exception as e:
                # NOTE [ avoid handing ] After adding the shared memory mechanism, not only
                # the queue.Empty exception will occur here, but other exceptions will also
                # occur, such as mmap failure. If it is not handled here, it will hang.
                self._exit_thread_unexpectedly()
                logging.error(
                    "DataLoader reader thread failed to read data from the multiprocessing.Queue."
                )
                raise e

            if not self._thread_done_event.is_set():
                if tensor_list is not None:
                    try:
                        array = core.LoDTensorArray()
                        for tensor in tensor_list:
                            array.append(tensor)
                        if not self._blocking_queue.push(array):
                            self._blocking_queue.close()
                    except Exception as e:
                        self._exit_thread_unexpectedly()
                        raise e
                else:
                    self._exit_thread_expectedly()

    def _reader_thread_loop_for_singleprocess(self, legacy_expected_place):
        try:
            # See _DataLoaderIterSingleProcess._thread_loop() for why set expected place here.
            core.set_current_thread_name("Dataloader_" + str(id(self)))
            _set_expected_place(legacy_expected_place)

            for sample in self._batch_reader():
                array = core.LoDTensorArray()
                for item in sample:
                    if not isinstance(item, core.LoDTensor):
                        item = self._check_input_array(item)
                        tmp = core.LoDTensor()
                        tmp.set(item, core.CPUPlace())
                        item = tmp

                    array.append(item)

                if not self._blocking_queue.push(array):
                    break

            self._blocking_queue.close()
            self._thread = None
        except Exception as e:
            self._blocking_queue.kill()
            self._thread = None
            logging.warning(
                "DygraphDataLoader reader thread raised an exception."
            )
            raise e

    def set_sample_generator(
        self, reader, batch_size, drop_last=True, places=None
    ):
        assert batch_size > 0, "batch_size must be larger than 0"
        if isinstance(places, (list, tuple)):
            places = _get_paddle_place_list(places)
        else:
            places = _get_paddle_place(places)
        self.set_sample_list_generator(
            paddle.batch(reader, batch_size=batch_size, drop_last=drop_last),
            places=places,
        )
        return self

    def set_sample_list_generator(self, reader, places=None):
        if isinstance(places, (list, tuple)):
            places = _get_paddle_place_list(places)
        else:
            places = _get_paddle_place(places)

        def __batch_reader_impl__():
            for batch in reader():
                slots = []
                for items in batch:
                    for i, item in enumerate(items):
                        if len(slots) < len(items):
                            slots.append([item])
                        else:
                            slots[i].append(item)
                yield slots

        self.set_batch_generator(__batch_reader_impl__, places)
        return self

    def set_batch_generator(self, reader, places=None):
        if isinstance(places, (list, tuple)):
            places = _get_paddle_place_list(places)
        else:
            places = _get_paddle_place(places)
        self._batch_reader = reader
        if places is None:
            places = _current_expected_place()
        self._places = _convert_places(places)
        assert (
            len(self._places) == 1
        ), "Number of places must be 1 in imperative mode"
        return self


class GeneratorLoader(DataLoaderBase):
    def __init__(
        self,
        feed_list=None,
        capacity=None,
        use_double_buffer=True,
        iterable=True,
        return_list=False,
        drop_last=True,
    ):
        self._tensor_reader = None
        self._places = None
        self._thread = None
        self._queue = None
        self._feed_list = feed_list
        self._exited = False
        self._drop_last = drop_last
        self._keep_order = keep_data_loader_order()
        if not capacity:
            raise ValueError("Please give value to capacity.")
        self._iterable = iterable
        self._return_list = return_list
        if not self._feed_list:
            raise Exception("Feed list must be given under static graph mode.")
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
        self._var_names = [v.name for v in self._feed_list]
        self._shapes = [v.shape for v in self._feed_list]
        if in_pir_mode():
            self._dtypes = [
                datatype_to_vartype[v.dtype] for v in self._feed_list
            ]
            self._need_check_feed = [False for v in self._feed_list]
        else:
            self._dtypes = [v.dtype for v in self._feed_list]
            self._need_check_feed = [
                v.desc.need_check_feed() for v in self._feed_list
            ]
        self._queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._capacity, self._keep_order
        )
        self._reader = None
        self._reader = core.create_py_reader(
            self.queue,
            self._var_names,
            self._shapes,
            self._dtypes,
            self._need_check_feed,
            self._places,
            self._use_double_buffer,
            self._drop_last,
            False,
        )

    def _init_non_iterable(self):
        lod_levels = []
        dtypes = []
        shape_concat = []
        ranks = []
        shapes = []
        need_check_feed = []

        for feed_data in self._feed_list:
            dtypes.append(feed_data.dtype)
            shape_concat.extend(feed_data.shape)
            ranks.append(len(feed_data.shape))
            shapes.append(feed_data.shape)
            lod_levels.append(feed_data.lod_level)
            if in_pir_mode():
                need_check_feed.append(0)
            else:
                need_check_feed.append(int(feed_data.desc.need_check_feed()))

        queue_name = data_loader_unique_name_generator(
            'lod_tensor_blocking_queue'
        )
        reader_name = data_loader_unique_name_generator('create_py_reader')
        double_buffer_name = data_loader_unique_name_generator('double_buffer')

        var = global_scope().var(queue_name)
        self._queue = core.init_lod_tensor_blocking_queue(
            var, self._capacity, self._keep_order
        )

        if self._keep_order:
            block = default_main_program().current_block()
        else:
            block = default_startup_program().current_block()

        reader_var = block.create_var(name=reader_name)

        dtype_int = [int(t) for t in dtypes]
        block.append_op(
            type='create_py_reader',
            inputs={'blocking_queue': [queue_name]},
            outputs={'Out': [reader_var]},
            attrs={
                'shape_concat': shape_concat,
                'lod_levels': lod_levels,
                'dtypes': dtype_int,
                'need_check_feed': need_check_feed,
                'ranks': ranks,
            },
        )

        reader_var.desc.set_dtypes(dtypes)
        reader_var.persistable = True
        reader_var.stop_gradient = True

        if self._keep_order:
            main_prog_var = reader_var
            reader = main_prog_var
            reader.reset = self._queue.reset
        else:
            main_prog_var = _copy_reader_var_(
                default_main_program().current_block(), reader_var
            )

            main_prog_var.stop_gradient = True
            main_prog_var.persistable = True

            reader = monkey_patch_reader_methods(main_prog_var)

        if self._use_double_buffer:
            double_buffer_reader = __create_unshared_decorated_reader__(
                'create_double_buffer_reader',
                reader,
                {},
                name=double_buffer_name,
            )
            # we return a double buffer reader. However, the reset method comes from
            # py_reader.
            double_buffer_reader.reset = reader.reset
            reader = double_buffer_reader

        self._reader = reader

        default_main_program().current_block().append_op(
            type='read',
            inputs={'Reader': [self._reader]},
            outputs={'Out': self._feed_list},
            attrs={'drop_last': self._drop_last},
        )

    @property
    def queue(self):
        return self._queue

    @property
    def iterable(self):
        return self._iterable

    def __iter__(self):
        assert self.iterable, "DataLoader is not iterable"
        assert (
            self._tensor_reader is not None
        ), "Data source of DataLoader has not set yet"

        self._init_iterable()
        self._start()
        return self

    def __next__(self):
        try:
            if self._return_list:
                data = self._reader.read_next_list()
                for i in range(len(data)):
                    data[i] = data[i]._move_to_list()
                return data
            else:
                return self._reader.read_next()
        except StopIteration:
            self._queue.close()
            self._reset()
            raise

    def start(self):
        assert (
            not self._iterable
        ), "start() cannot be called when DataLoader is iterable"
        self._start()

    def reset(self):
        assert (
            not self._iterable
        ), "reset() cannot be called when DataLoader is iterable"
        self._reset()

    def _start(self):
        def __thread_main__(legacy_expected_place):
            try:
                # See _DataLoaderIterSingleProcess._thread_loop() for why set expected place here.
                core.set_current_thread_name("Dataloader_" + str(id(self)))
                _set_expected_place(legacy_expected_place)

                while not self._queue.wait_for_inited(1):
                    if self._exited:
                        return

                for tensors in self._tensor_reader():
                    array = core.LoDTensorArray()
                    for item in tensors:
                        if not isinstance(item, core.LoDTensor):
                            item = self._check_input_array(item)
                            tmp = core.LoDTensor()
                            tmp.set(item, core.CPUPlace())
                            item = tmp

                        array.append(item)

                    if not self._queue.push(array):
                        break

                self._queue.close()
                self._thread = None
            except Exception as e:
                self._queue.kill()
                self._thread = None
                logging.warning('Your reader has raised an exception!')
                raise e

        self._thread = threading.Thread(
            target=__thread_main__, args=(_current_expected_place(),)
        )
        self._thread.daemon = True
        self._thread.start()

    def _reset(self):
        self._queue.close()
        self._exited = True
        thread = self._thread
        if thread is not None:
            thread.join()

        self._exited = False
        self._reader.reset()

    def set_sample_generator(
        self, reader, batch_size, drop_last=True, places=None
    ):
        assert batch_size > 0, "batch_size must be larger than 0"
        if isinstance(places, (list, tuple)):
            places = _get_paddle_place_list(places)
        else:
            places = _get_paddle_place(places)
        has_lod = False
        for f in self._feed_list:
            if f.lod_level != 0:
                has_lod = True
                break

        if has_lod:
            self.set_sample_list_generator(
                paddle.batch(
                    reader, batch_size=batch_size, drop_last=drop_last
                ),
                places=places,
            )
        else:
            reader = BatchedTensorProvider(
                feed_list=self._feed_list,
                place=core.CPUPlace(),
                batch_size=batch_size,
                generator=reader,
                drop_last=drop_last,
            )
            self.set_batch_generator(reader, places=places)
        return self

    def set_sample_list_generator(self, reader, places=None):
        if isinstance(places, (list, tuple)):
            places = _get_paddle_place_list(places)
        else:
            places = _get_paddle_place(places)
        with program_guard(Program(), Program()):
            feeder = DataFeeder(
                feed_list=self._feed_list, place=core.CPUPlace()
            )

            def decorate_reader():
                for item in reader():
                    yield feeder.feed(item)

            paddle_reader = decorate_reader

        def __tensor_reader_impl__():
            for slots in paddle_reader():
                yield [slots[var.name] for var in self._feed_list]

        self.set_batch_generator(__tensor_reader_impl__, places)
        return self

    def set_batch_generator(self, reader, places=None):
        if isinstance(places, (list, tuple)):
            places = _get_paddle_place_list(places)
        else:
            places = _get_paddle_place(places)
        self._tensor_reader = reader
        if self._iterable:
            assert (
                places is not None
            ), "Places cannot be None when DataLoader is iterable"
            self._places = _convert_places(places)
        else:
            if places is not None:
                logging.info(
                    'places would be omitted when DataLoader is not iterable'
                )
        return self


class PyReader(DataLoaderBase):
    r"""
    Create a reader object for data feeding in Python.
    Data would be prefetched using Python thread and be pushed
    into a queue asynchronously. Data in the queue would be extracted
    automatically when `Executor.run(...)` is called.

    Args:
        feed_list (list(Variable)|tuple(Variable)): feed variable list.
            The variables should be created by :code:`paddle.static.data()`.
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
            the name of each fed variables. If return_list=True, the
            return value on each device would be a list(LoDTensor). It is
            recommended to use return_list=False in static graph mode and
            use return_list=True in dygraph mode.

    Returns:
        the created reader object.

    Return type:
        reader(Reader)

    Examples:

        1. If iterable = False, the created PyReader object is almost the
           same as :code:`base.layers.py_reader()`. Operators would be
           inserted into the program. User should call :code:`start()`
           before each epoch and catch :code:`base.core.EOFException`
           thrown by :code:`Executor.run()` when epoch ends. Once the
           exception is caught, user should call :code:`reset()` to reset
           the reader manually.

        .. code-block:: python
            :name: example_1

            >>> import paddle
            >>> import paddle.base as base
            >>> import numpy as np

            >>> paddle.enable_static()

            >>> EPOCH_NUM = 3
            >>> ITER_NUM = 5
            >>> BATCH_SIZE = 3

            >>> def network(image, label):
            ...     # User-defined network, here is an example of softmax regression.
            ...     predict = paddle.static.nn.fc(x=image, size=10, activation='softmax')
            ...     return paddle.nn.functional.cross_entropy(
            ...         input=predict, label=label,
            ...         reduction='none', use_softmax=False
            ...     )

            >>> def reader_creator_random_image_and_label(height, width):
            ...     def reader():
            ...         for i in range(ITER_NUM):
            ...             fake_image = np.random.uniform(low=0,
            ...                                             high=255,
            ...                                             size=[height, width])
            ...             fake_label = np.ones([1])
            ...             yield fake_image, fake_label
            ...     return reader

            >>> image = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
            >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')

            >>> reader = base.io.PyReader(feed_list=[image, label],
            ...                             capacity=4,
            ...                             iterable=False)

            >>> user_defined_reader = reader_creator_random_image_and_label(784, 784)
            >>> reader.decorate_sample_list_generator(
            ...     paddle.batch(user_defined_reader, batch_size=BATCH_SIZE))
            >>> loss = network(image, label)
            >>> executor = base.Executor(base.CPUPlace())
            >>> executor.run(base.default_startup_program())
            >>> for i in range(EPOCH_NUM):
            ...     reader.start()
            ...     while True:
            ...         try:
            ...             executor.run(feed=None)
            ...         except base.core.EOFException:
            ...             reader.reset()
            ...             break

        2. If iterable=True, the created PyReader object is decoupled with
           the program. No operator would be inserted into the program.
           In this case, the created reader is a Python generator, which
           is iterable. User should feed the data yielded from PyReader
           object into :code:`Executor.run(feed=...)`.

        .. code-block:: python
            :name: example_2

            >>> import paddle
            >>> import paddle.base as base
            >>> import numpy as np

            >>> paddle.enable_static()

            >>> EPOCH_NUM = 3
            >>> ITER_NUM = 5
            >>> BATCH_SIZE = 10

            >>> def network(image, label):
            ...     # User-defined network, here is an example of softmax regression.
            ...     predict = paddle.static.nn.fc(x=image, size=10, activation='softmax')
            ...     return paddle.nn.functional.cross_entropy(
            ...         input=predict, label=label,
            ...         reduction='none', use_softmax=False
            ...     )

            >>> def reader_creator_random_image(height, width):
            ...     def reader():
            ...         for i in range(ITER_NUM):
            ...             fake_image = np.random.uniform(low=0, high=255, size=[height, width])
            ...             fake_label = np.ones([1])
            ...             yield fake_image, fake_label
            ...     return reader

            >>> image = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
            >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            >>> reader = base.io.PyReader(feed_list=[image, label], capacity=4, iterable=True, return_list=False)

            >>> user_defined_reader = reader_creator_random_image(784, 784)
            >>> reader.decorate_sample_list_generator(
            ...     paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
            ...         base.core.CPUPlace())

            >>> loss = network(image, label)
            >>> executor = base.Executor(base.CPUPlace())
            >>> executor.run(base.default_startup_program())

            >>> for _ in range(EPOCH_NUM):
            ...     for data in reader():
            ...         executor.run(feed=data, fetch_list=[loss])

        3. If return_list=True, the return values would be presented as list instead of dict.
           This is usually used in dygraph mode.

        .. code-block:: python
            :name: example_3

            >>> import paddle
            >>> import paddle.base as base
            >>> import numpy as np

            >>> ITER_NUM = 5
            >>> BATCH_SIZE = 10

            >>> def reader_creator_random_image(height, width):
            ...     def reader():
            ...         for i in range(ITER_NUM):
            ...             yield np.random.uniform(low=0, high=255, size=[height, width]), \
            ...                 np.random.random_integers(low=0, high=9, size=[1])
            ...     return reader

            >>> place = base.CPUPlace()
            >>> with base.dygraph.guard(place):
            ...     py_reader = base.io.PyReader(capacity=2, return_list=True)
            ...     user_defined_reader = reader_creator_random_image(784, 784)
            ...     py_reader.decorate_sample_list_generator(
            ...         paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
            ...         place)
            ...     for image, label in py_reader():
            ...         relu = paddle.nn.functional.relu(image)
    """

    def __init__(
        self,
        feed_list=None,
        capacity=None,
        use_double_buffer=True,
        iterable=True,
        return_list=False,
    ):
        self._loader = DataLoader.from_generator(
            feed_list, capacity, use_double_buffer, iterable, return_list
        )

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

                >>> import paddle
                >>> import paddle.base as base
                >>> import numpy as np

                >>> paddle.enable_static()

                >>> BATCH_SIZE = 10

                >>> def generator():
                ...     for i in range(5):
                ...         yield np.random.uniform(low=0, high=255, size=[784, 784]),

                >>> image = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
                >>> reader = base.io.PyReader(feed_list=[image], capacity=4, iterable=False)
                >>> reader.decorate_sample_list_generator(
                ...     paddle.batch(generator, batch_size=BATCH_SIZE))

                >>> executor = base.Executor(base.CPUPlace())
                >>> executor.run(base.default_startup_program())
                >>> for i in range(3):
                ...     reader.start()
                ...     while True:
                ...         try:
                ...             executor.run(feed=None)
                ...         except base.core.EOFException:
                ...             reader.reset()
                ...             break
        '''
        self._loader.start()

    def reset(self):
        '''
        Reset the reader object when :code:`base.core.EOFException` raises.
        Can only call when the reader object is not iterable.

        Example:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.base as base
                >>> import numpy as np

                >>> paddle.enable_static()

                >>> BATCH_SIZE = 10

                >>> def generator():
                ...     for i in range(5):
                ...         yield np.random.uniform(low=0, high=255, size=[784, 784]),

                >>> image = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
                >>> reader = base.io.PyReader(feed_list=[image], capacity=4, iterable=False)
                >>> reader.decorate_sample_list_generator(
                ...     paddle.batch(generator, batch_size=BATCH_SIZE))

                >>> executor = base.Executor(base.CPUPlace())
                >>> executor.run(base.default_startup_program())
                >>> for i in range(3):
                ...     reader.start()
                ...     while True:
                ...         try:
                ...             executor.run(feed=None)
                ...         except base.core.EOFException:
                ...             reader.reset()
                ...             break
        '''
        self._loader.reset()

    def decorate_sample_generator(
        self, sample_generator, batch_size, drop_last=True, places=None
    ):
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

                >>> import paddle
                >>> import paddle.base as base
                >>> import numpy as np

                >>> paddle.enable_static()

                >>> EPOCH_NUM = 3
                >>> ITER_NUM = 15
                >>> BATCH_SIZE = 3

                >>> def network(image, label):
                ...     # User-defined network, here is an example of softmax regression.
                ...     predict = paddle.static.nn.fc(x=image, size=10, activation='softmax')
                ...     return paddle.nn.functional.cross_entropy(
                ...         input=predict, label=label,
                ...         reduction='none', use_softmax=False
                ...     )

                >>> def random_image_and_label_generator(height, width):
                ...     def generator():
                ...         for i in range(ITER_NUM):
                ...             fake_image = np.random.uniform(low=0,
                ...                                             high=255,
                ...                                             size=[height, width])
                ...             fake_label = np.array([1])
                ...             yield fake_image, fake_label
                ...     return generator

                >>> image = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
                >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
                >>> reader = base.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)

                >>> user_defined_generator = random_image_and_label_generator(784, 784)
                >>> reader.decorate_sample_generator(user_defined_generator,
                ...                                     batch_size=BATCH_SIZE,
                ...                                     places=[base.CPUPlace()])
                >>> loss = network(image, label)
                >>> executor = base.Executor(base.CPUPlace())
                >>> executor.run(base.default_startup_program())

                >>> for _ in range(EPOCH_NUM):
                ...     for data in reader():
                ...         executor.run(feed=data, fetch_list=[loss])
        '''
        self._loader.set_sample_generator(
            sample_generator, batch_size, drop_last, places
        )

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

                >>> import paddle
                >>> import paddle.base as base
                >>> import numpy as np

                >>> paddle.enable_static()

                >>> EPOCH_NUM = 3
                >>> ITER_NUM = 15
                >>> BATCH_SIZE = 3

                >>> def network(image, label):
                ...     # User-defined network, here is an example of softmax regression.
                ...     predict = paddle.static.nn.fc(x=image, size=10, activation='softmax')
                ...     return paddle.nn.functional.cross_entropy(
                ...         input=predict, label=label,
                ...         reduction='none', use_softmax=False
                ...     )

                >>> def random_image_and_label_generator(height, width):
                ...     def generator():
                ...         for i in range(ITER_NUM):
                ...             fake_image = np.random.uniform(low=0,
                ...                                             high=255,
                ...                                             size=[height, width])
                ...             fake_label = np.ones([1])
                ...             yield fake_image, fake_label
                ...     return generator

                >>> image = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
                >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
                >>> reader = base.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)

                >>> user_defined_generator = random_image_and_label_generator(784, 784)
                >>> reader.decorate_sample_list_generator(
                ...     paddle.batch(user_defined_generator, batch_size=BATCH_SIZE),
                ...     base.core.CPUPlace())

                >>> loss = network(image, label)
                >>> executor = base.Executor(base.core.CPUPlace())
                >>> executor.run(base.default_startup_program())

                >>> for _ in range(EPOCH_NUM):
                ...     for data in reader():
                ...         executor.run(feed=data, fetch_list=[loss])
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

                >>> import paddle
                >>> import paddle.base as base
                >>> import numpy as np

                >>> paddle.enable_static()

                >>> EPOCH_NUM = 3
                >>> ITER_NUM = 15
                >>> BATCH_SIZE = 3

                >>> def network(image, label):
                ...     # User-defined network, here is an example of softmax regression.
                ...     predict = paddle.static.nn.fc(x=image, size=10, activation='softmax')
                ...     return paddle.nn.functional.cross_entropy(
                ...         input=predict, label=label,
                ...         reduction='none', use_softmax=False
                ...     )

                >>> def random_image_and_label_generator(height, width):
                ...     def generator():
                ...         for i in range(ITER_NUM):
                ...             batch_image = np.random.uniform(low=0,
                ...                                             high=255,
                ...                                             size=[BATCH_SIZE, height, width])
                ...             batch_label = np.ones([BATCH_SIZE, 1])
                ...             batch_image = batch_image.astype('float32')
                ...             batch_label = batch_label.astype('int64')
                ...             yield batch_image, batch_label
                ...     return generator

                >>> image = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
                >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
                >>> reader = base.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)

                >>> user_defined_generator = random_image_and_label_generator(784, 784)
                >>> reader.decorate_batch_generator(user_defined_generator, base.CPUPlace())

                >>> loss = network(image, label)
                >>> executor = base.Executor(base.CPUPlace())
                >>> executor.run(base.default_startup_program())

                >>> for _ in range(EPOCH_NUM):
                ...     for data in reader():
                ...         executor.run(feed=data, fetch_list=[loss])
        '''
        self._loader.set_batch_generator(reader, places)


class DatasetLoader(DataLoaderBase):
    def __init__(self, dataset, places, drop_last):
        assert isinstance(
            dataset, paddle.distributed.fleet.dataset.DatasetBase
        ), "dataset must be type of DatasetBase"
        assert (
            not in_dygraph_mode()
        ), "DatasetLoader is not supported in dygraph mode yet"
        if isinstance(places, (list, tuple)):
            places = _get_paddle_place_list(places)
        else:
            places = _get_paddle_place(places)

        thread_num = len(places)

        assert (
            len(dataset.filelist) >= thread_num
        ), f"Filelist number of dataset {len(dataset.filelist)} must be not less than place number {thread_num}"

        if dataset.thread_num != 0 and dataset.thread_num != thread_num:
            logging.warn(
                f'thread_num {dataset.thread_num} which is set in Dataset is ignored'
            )

        dataset._set_thread(thread_num)

        if (
            isinstance(
                dataset, paddle.distributed.fleet.dataset.InMemoryDataset
            )
            and dataset.queue_num > thread_num
        ):
            logging.warn(
                f"queue_num {dataset.queue_num} which is set in Dataset is ignored"
            )
            dataset._set_queue_num(thread_num)

        self._dataset = dataset
        use_slots = [
            slot.name
            for slot in dataset.proto_desc.multi_slot_desc.slots
            if slot.is_used
        ]

        self._iterable_dataset = core.IterableDatasetWrapper(
            dataset.dataset,
            use_slots,
            _convert_places(places),
            dataset.proto_desc.batch_size,
            drop_last,
        )

    def __iter__(self):
        self._dataset._finish_to_run()
        self._dataset._prepare_to_run()
        self._iterable_dataset._start()
        return self

    def __next__(self):
        return self._iterable_dataset._next()
