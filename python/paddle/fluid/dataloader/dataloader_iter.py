# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import six
import sys
import time
import signal
import logging
import itertools
import threading
import numpy as np
import multiprocessing

# NOTE: queue has a different name in python2 and python3
if six.PY2:
    import Queue as queue
else:
    import queue

from .. import core
from ..framework import in_dygraph_mode
from ..multiprocess_utils import multiprocess_queue_set, CleanupFuncRegistrar, _cleanup_mmap

# multi-process worker check indices queue interval, avoid
# hanging in subprocess data loading
MP_INDICES_CHECK_INTERVAL = 5

__all__ = ['_DataLoaderIterSingleProcess', '_DataLoaderIterMultiProcess']


class _DataLoaderIterBase(object):
    """
    Iterator implement of DataLoader, will load and feed mini-batch
    data by setting in given dataloader.

    Args:
        loader(instance of DataLoader): instance of `fluid.io.DataLoader`
    """

    def __init__(self, loader):
        self._dataset = loader.dataset
        self._feed_list = loader.feed_list or []
        self._places = loader.places
        self._return_list = loader.return_list
        self._batch_sampler = loader.batch_sampler
        self._sampler_iter = iter(loader.batch_sampler)
        self._collate_fn = loader.collate_fn
        self._num_workers = loader.num_workers
        self._use_buffer_reader = loader.use_buffer_reader
        self._timeout = loader.timeout
        self._worker_init_fn = loader.worker_init_fn

        # LoDTensorBlockingQueue instance for create_py_reader and a thread
        # to put mini-batch data to self._blocking_queue, mini-batch data
        # will be get from:
        # 1. multi-process mode: get data from workers' result queue
        # 2. single-process mode: read mini-batch data in main process
        self._blocking_queue = None
        self._thread = None
        self._thread_done_event = threading.Event()

    def _wait_thread_ends(self):
        if self._thread is not None:
            self._blocking_queue.close()
            self._thread_done_event.set()
            self._thread = None

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__next__', self.__class__.__name__))

    # python2 compatibility
    def next(self):
        return self.__next__()

    # following method is a remind for users initialize DataLoader
    # with __init__ but call set_xxx(...) wrongly
    def set_sample_generator(*args, **kwargs):
        raise Exception("DataLoader is initialize, can be used as an " \
                        "iterater, should not call set_sample_generator")

    def set_sample_list_generator(*args, **kwargs):
        raise Exception("DataLoader is initialize, can be used as an " \
                        "iterater, should not call set_sample_list_generator")

    def set_batch_generator(*args, **kwargs):
        raise Exception("DataLoader is initialize, can be used as an " \
                        "iterater, should not call set_batch_generator")


class _DataLoaderIterSingleProcess(_DataLoaderIterBase):
    """
    Single process implement of DataLoaderIter, loading data from
    loader.data in main process
    """

    def __init__(self, loader):
        super(_DataLoaderIterSingleProcess, self).__init__(loader)

        # NOTE: len(self._places) batch data compose as an output
        # iteration, set blocking_queue can cache 2 iteration datas
        # at most here
        self._blocking_queue_capacity = 2 * len(self._places)

        self._init_thread()

    def _init_thread(self):
        self._wait_thread_ends()
        self._var_names = [v.name for v in self._feed_list]
        self._shapes = [v.shape for v in self._feed_list]
        self._dtypes = [v.dtype for v in self._feed_list]
        self._need_check_feed = [
            v.desc.need_check_feed() for v in self._feed_list
        ]
        self._blocking_queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._blocking_queue_capacity)
        self._reader = core.create_py_reader(
            self._blocking_queue, self._var_names, self._shapes, self._dtypes,
            self._need_check_feed, self._places, self._use_buffer_reader)

        self._thread = threading.Thread(target=self._thread_loop)
        self._thread.daemon = True
        self._thread.start()

    def _thread_loop(self):
        try:
            for indices in self._sampler_iter:
                # read data from dataset in mini-batch
                batch = [self._dataset[i] for i in indices]
                if self._collate_fn is not None:
                    batch = _collate_fn(batch)

                # batch each field
                slots = []
                for items in batch:
                    for i, item in enumerate(items):
                        if len(slots) < len(items):
                            slots.append([item])
                        else:
                            slots[i].append(item)

                # pack as LoDTensorArray
                array = core.LoDTensorArray()
                for slot in slots:
                    if not isinstance(slot, core.LoDTensor):
                        self._check_input_array(slot)
                        tmp = core.LoDTensor()
                        tmp.set(slot, core.CPUPlace())
                        slot = tmp

                    array.append(slot)

                if not self._blocking_queue.push(array):
                    break

            self._blocking_queue.close()
            self._thread = None
        except Exception:
            self._blocking_queue.kill()
            self._thread = None
            logging.warning("DataLoader reader thread raised an exception.")
            six.reraise(*sys.exc_info())

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

    def __next__(self):
        try:
            if in_dygraph_mode():
                return self._reader.read_next_var_list()
            else:
                if self._return_list:
                    return self._reader.read_next_list()
                else:
                    return self._reader.read_next()
        except StopIteration:
            self._reader.reset()
            six.reraise(*sys.exc_info())


class _DataLoaderIterMultiProcess(_DataLoaderIterBase):
    def __init__(self, loader):
        super(_DataLoaderIterMultiProcess, self).__init__(loader)

        assert self._num_workers > 0,  "Multi-process DataLoader " \
                    "invalid num_workers({})".format(self._num_workers)

        # subprocess wrokers' result queue
        self._data_queue = None

        # data get from _data_queue will be reordered by _rcvd_idx
        # for data order keeping, data index not equal _rcvd_idx 
        # will be cached in _reorder_dict
        self._send_idx = 0
        self._rcvd_idx = 0
        self._batches_outstanding = 0
        self._reorder_dict = {}

        # indices outstand as _outstanding_capacity at first, and
        # blocking_queue capacity is also _outstanding_capacity.
        # _outstanding_capacity here to make sure each indices_queue
        # has at least 2 indices, and outstanding batch cached
        # output data for at least 2 iterations(Note that len(_places)
        # batches will be composed as an iteration output)
        self._outstanding_capacity = 2 * max(self._num_workers,
                                             len(self._places))

        self._init_workers()
        self._init_thread()

        for _ in range(self._outstanding_capacity):
            self._try_put_indices()

    def _init_workers(self):
        # multiprocess worker and indice queue list initial as empty
        self._workers = []
        self._indices_queues = []
        self._workers_idx_cycle = itertools.cycle(range(self._num_workers))

        # create data_queue for workers
        self._data_queue = multiprocessing.Queue()
        global multiprocess_queue_set
        multiprocess_queue_set.add(self._data_queue)

        # event for workers and thread, thread event is only need 
        # in multi-processing mode
        self._workers_done_event = multiprocessing.Event()
        self._thread_done_event = threading.Event()

        for i in range(self._num_workers):
            indices_queue = multiprocessing.Queue()
            self._indices_queues.append(indices_queue)
            worker = multiprocessing.Process(
                target=self._worker_loop,
                args=(self._dataset, indices_queue, self._data_queue,
                      self._workers_done_event, self._collate_fn,
                      self._worker_init_fn, i))
            worker.daemon = True
            worker.start()
            self._set_worker_signal_handler(worker)
            self._workers.append(worker)

    def _set_worker_signal_handler(self, worker):
        core._set_process_pid(id(worker), worker.pid)
        current_handler = signal.getsignal(signal.SIGCHLD)
        if not callable(current_handler):
            current_handler = None

        def __handler__(signum, frame):
            # NOTE: Here the signum is SIGDHLD, when the child process
            # exits, this handler will be called whenever the child
            # process exits normally or abnormally.
            core._throw_error_if_process_failed()
            if current_handler is not None:
                current_handler(signum, frame)

        signal.signal(signal.SIGCHLD, __handler__)

    def _clear_and_remove_data_queue(self):
        if self._data_queue is not None:
            while True:
                try:
                    self._data_queue.get_nowait()
                except queue.Empty:
                    self._data_queue.cancel_join_thread()
                    self._data_queue.close()
                    break
            global multiprocess_queue_set
            multiprocess_queue_set.remove(self._data_queue)

    def _init_thread(self):
        self._wait_thread_ends()
        self._var_names = [v.name for v in self._feed_list]
        self._shapes = [v.shape for v in self._feed_list]
        self._dtypes = [v.dtype for v in self._feed_list]
        self._need_check_feed = [
            v.desc.need_check_feed() for v in self._feed_list
        ]
        self._blocking_queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._outstanding_capacity)
        self._reader = core.create_py_reader(
            self._blocking_queue, self._var_names, self._shapes, self._dtypes,
            self._need_check_feed, self._places, self._use_buffer_reader)

        self._thread_done_event = threading.Event()
        self._thread = threading.Thread(target=self._thread_loop)
        self._thread.daemon = True
        self._thread.start()

    def _wait_workers_ends(self):
        # 1. set _workers_done_event before close queue and end workers
        self._workers_done_event.set()

        # 2. put None to indices_queue to exit workers, join workers
        #    and close indices_queue
        for worker, queue in zip(self._workers, self._indices_queues):
            # put None to indices queue, worker will exit if read None
            queue.put(None)

            # wait worker joined and erase pid
            worker.join()
            core._erase_process_pid(id(worker))

            # close indices_queue after worker exit
            queue.cancel_join_thread()
            queue.close()

        # 3. objects in _data_queue are on shared memory, need to read
        #    them out and remove from global set
        self._clear_and_remove_data_queue()

    def _exit_thread_expectedly(self):
        self._thread_done_event.set()
        self._blocking_queue.close()

    def _exit_thread_unexpectedly(self):
        self._thread_done_event.set()
        self._blocking_queue.kill()
        logging.error("DataLoader reader thread raised an exception!")

    def _worker_loop(self, dataset, indices_queue, out_queue, done_event,
                     collate_fn, init_fn, worker_id):
        try:
            # set signal handler
            core._set_process_signal_handler()

            # NOTE: [ mmap files clear ] When the child process exits unexpectedly,
            # some shared memory objects may have been applied for but have not yet
            # been put into the inter-process Queue. This part of the object needs
            # to be cleaned up when the process ends.
            CleanupFuncRegistrar.register(_cleanup_mmap)

            init_exception = None
            if init_fn is not None:
                try:
                    init_fn(worker_id)
                except:
                    init_exception = Exception("init_fn failed in worker {}: " \
                                         "{}".format(worker_id, sys.exc_info()))

            while True:
                try:
                    data = indices_queue.get(MP_INDICES_CHECK_INTERVAL)
                except queue.Empty:
                    continue

                # None as final signal, so worker event should be set
                if data is None:
                    assert done_event.is_set()
                    return
                # If worker done event is set but get still get data in
                # indices_queue, remaining data should be get and skipped.
                elif done_event.is_set():
                    continue

                idx, indices = data
                try:
                    if init_exception is not None:
                        batch = init_exception
                        init_exception = None
                    else:
                        batch = [dataset[i] for i in indices]
                        if self._collate_fn is not None:
                            batch = self._collate_fn(batch)
                except:
                    out_queue.put((idx, Exception("Get data failed in worker {}: " \
                                   "{}".format(worker_id, sys.exc_info()))))
                else:
                    # copy batch data to shared memory
                    slots = []
                    for items in batch:
                        for i, item in enumerate(items):
                            if len(slots) < len(items):
                                slots.append([item])
                            else:
                                slots[i].append(item)

                    tensor_list = core._convert_to_tensor_list(slots)
                    out_queue.put((idx, tensor_list))
                    core._remove_tensor_list_mmap_fds(tensor_list)
        except KeyboardInterrupt:
            # NOTE: Main process will raise KeyboardInterrupt anyways, ignore it in child process
            pass
        except:
            six.reraise(*sys.exc_info())

        if self._workers_done_event.is_set():
            _cleanup_mmap()

    def _thread_loop(self):
        while not self._thread_done_event.is_set():
            data = self._get_data()
            if data is None:
                continue
            idx, batch = data
            self._rcvd_idx += 1
            if not self._thread_done_event.is_set():
                if batch is not None:
                    try:
                        array = core.LoDTensorArray()
                        for tensor in batch:
                            array.append(tensor)
                        if not self._blocking_queue.push(array):
                            self._blocking_queue.close()
                    except:
                        self._exit_thread_unexpectedly()
                        six.reraise(*sys.exc_info())
                else:
                    self._exit_thread_expectedly()

    def _get_data(self):
        if self._rcvd_idx in self._reorder_dict.keys():
            return self._reorder_dict.pop(self._rcvd_idx)

        while not self._thread_done_event.is_set():
            try:
                # NOTE: [ avoid hanging ] Even with carefully designed data dependencies 
                # (i.e., a put() always corresponding to a get()), hanging on get() can 
                # still happen when data in queue is corrupted (e.g., due to 
                # Queue.cancel_join_thread or unexpected exit). So we set a timeout whenever 
                # we try to get data from `data_queue`
                # NOTE: [ avoid failed quickly ] default timeout setting QUEUE_GET_TIMEOUT
                # is relatively long, currently it is 60 seconds, because in some models,
                # if the reader child process starts with a heavy burden, the child process
                # has no enough time to put the data in the queue when the main process
                # start trying to get data from queue. At this time, the child thread needs
                # to wait slightly longer
                if self._timeout > 0:
                    data = self._data_queue.get(timeout=self._timeout)
                else:
                    data = self._data_queue.get()
            except queue.Empty:
                # when data drained, _data_queue.get may start before _thread_done_event set
                # when get queue.Empty, check _thread_done_event is set or python exit, end
                # thread quietly, otherwise raise exception
                if self._thread_done_event.is_set():
                    return
                self._exit_thread_unexpectedly()
                logging.error("DataLoader reader thread read workers' result queue get " \
                              "Empty before reading finished.")
                six.reraise(*sys.exc_info())
            except Exception as e:
                # NOTE [ avoid handing ] After adding the shared memory mechanism, not only
                # the queue. Empty exception will occur here, but other exceptions will also
                # occur, such as mmap failure. If it is not handled here, it will hang.
                self._exit_thread_unexpectedly()
                logging.error("DataLoader reader thread failed({}) to read data from " \
                              "workers' result queue.".format(e))
                six.reraise(*sys.exc_info())
            else:
                idx, batch = data
                if isinstance(batch, Exception):
                    raise batch
                if idx == self._rcvd_idx:
                    return data
                else:
                    self._reorder_dict[idx] = data
                    continue

    def _try_put_indices(self):
        assert self._send_idx - self._rcvd_idx <= self._outstanding_capacity, \
                    "too many indices have been put to queue"
        try:
            indices = next(self._sampler_iter)
        except StopIteration:
            return

        worker_idx = next(self._workers_idx_cycle)
        self._indices_queues[worker_idx].put((self._send_idx, indices))
        self._batches_outstanding += 1
        self._send_idx += 1

    def __next__(self):
        try:
            # _batches_outstanding here record the total batch data number
            # in 'from after _try_put_indices to beforeoutput data', this
            # value should be _outstanding_capacity if data is not drained,
            # if _batches_outstanding is less than _places number, there are
            # no enough data to generate next output, close blocking_queue and
            # set _thread_done_event here, py_reader will raise StopIteration,
            # end workers and indices_queues in StopIteration handling
            if self._batches_outstanding < len(self._places):
                # datas in workers should be all processed when starting to
                # end this epoch, if there are still data in workers, some
                # exceptions may occured in workers, log details here
                if self._send_idx != self._rcvd_idx:
                    worker_status = [w.is_alive() for w in self._workers]
                    indices_queue_lens = [
                        q.qsize() for q in self._indices_queues
                    ]
                    logging.warn("Data drained for outstanding batches({})"
                                 " < places num({}) in multiprocessing, except "
                                 "send_idx({}) == rcvd_idx({}) but not, status:"
                                 "\n  outstanding capacity: {}"
                                 "\n  reorder_dict indices: {}"
                                 "\n  blocking_queue size: {}"
                                 "\n  thread_event set: {}"
                                 "\n  worker_event set: {}"
                                 "\n  worker status: {}"
                                 "\n  indices_queue length: {}".format(
                                     self._batches_outstanding,
                                     len(self._places), self._send_idx,
                                     self._rcvd_idx, self._outstanding_capacity,
                                     self._reorder_dict.keys(),
                                     self._blocking_queue.size(),
                                     self._thread_done_event.is_set(),
                                     self._workers_done_event.is_set(
                                     ), worker_status, indices_queue_lens))
                self._thread_done_event.set()
                self._blocking_queue.close()

            if in_dygraph_mode():
                data = self._reader.read_next_var_list()
            else:
                if self._return_list:
                    data = self._reader.read_next_list()
                else:
                    data = self._reader.read_next()
            self._on_output_batch()
            return data
        except StopIteration:
            self._reader.reset()
            self._wait_workers_ends()
            six.reraise(*sys.exc_info())

    def _on_output_batch(self):
        for _ in range(len(self._places)):
            self._batches_outstanding -= 1
            self._try_put_indices()
