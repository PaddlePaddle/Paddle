# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import annotations

import itertools
import logging
import multiprocessing
import random
import sys
import warnings
from itertools import zip_longest
from queue import Queue
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    TypedDict,
    TypeVar,
    overload,
)

from typing_extensions import NotRequired, TypeAlias, Unpack

from paddle.base.reader import QUEUE_GET_TIMEOUT

if TYPE_CHECKING:
    from collections.abc import Sequence

    class _ComposeOptions(TypedDict):
        check_alignment: NotRequired[bool]


__all__ = []

# On macOS, the 'spawn' start method is now the default in Python3.8 multiprocessing,
# Paddle is currently unable to solve this, so forces the process to start using
# the 'fork' start method.
#
# TODO: This solution is not good, because the fork start method could lead to
# crashes of the subprocess. Figure out how to make 'spawn' work.
#
# For more details, please refer to
# https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
# https://bugs.python.org/issue33725
if sys.version_info >= (3, 8) and sys.platform == 'darwin':
    fork_context = multiprocessing.get_context('fork')
else:
    fork_context = multiprocessing

_T = TypeVar('_T')
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')
_T3 = TypeVar('_T3')
_T4 = TypeVar('_T4')
_U = TypeVar('_U')


_Reader: TypeAlias = Callable[[], Generator[_T, None, None]]


def cache(reader: _Reader[_T]) -> _Reader[_T]:
    """
    Cache the reader data into memory.

    Be careful that this method may take long time to process,
    and consume lots of memory. :code:`reader()` would only
    call once.

    Args:
        reader (generator): a reader object which yields
            data each time.

    Returns:
        generator: a decorated reader object which yields data from cached memory.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> def reader():
            ...     for i in range(3):
            ...         yield i
            ...
            >>> # All data is cached into memory
            >>> cached_reader = paddle.base.io.cache(reader)

            >>> for i in cached_reader():
            ...     print(i)
            0
            1
            2
    """
    all_data = tuple(reader())

    def __impl__() -> Generator[_T, None, None]:
        yield from all_data

    return __impl__


# A temporary solution like builtin map function.
# `Map` maybe the final solution in the future.
# See https://github.com/python/typing/issues/1383
@overload
def map_readers(
    func: Callable[[_T1], _U], reader1: _Reader[_T1], /
) -> _Reader[_U]: ...


@overload
def map_readers(
    func: Callable[[_T1, _T2], _U],
    reader1: _Reader[_T1],
    reader2: _Reader[_T2],
    /,
) -> _Reader[_U]: ...


@overload
def map_readers(
    func: Callable[[_T1, _T2, _T3], _U],
    reader1: _Reader[_T1],
    reader2: _Reader[_T2],
    reader3: _Reader[_T3],
    /,
) -> _Reader[_U]: ...


@overload
def map_readers(
    func: Callable[[_T1, _T2, _T3, _T4], _U],
    reader1: _Reader[_T1],
    reader2: _Reader[_T2],
    reader3: _Reader[_T3],
    reader4: _Reader[_T4],
    /,
) -> _Reader[_U]: ...


@overload
def map_readers(
    func: Callable[..., _U], *readers: _Reader[Any]
) -> _Reader[_U]: ...


def map_readers(func, *readers):
    """
    Creates a data reader that outputs return value of function using
    output of each data reader as arguments.

    If input readers output the following data entries: 2 3,
    and the input func is mul(x, y),
    the output of the resulted reader will be 6.


    Args:
        func: a function to read data and compute result, the output of this function
              will be set as the output of the resulted data reader.
        readers (Reader|list of Reader): list of readers whose outputs will be used as arguments of func.

    Returns:
        the resulted data reader (Reader)

    Examples:

        .. code-block:: python

            >>> import paddle.reader
            >>> d = {"h": 0, "i": 1}
            >>> def func(x):
            ...     return d[x]
            >>> def reader():
            ...     yield "h"
            ...     yield "i"
            >>> map_reader_result = paddle.reader.map_readers(func, reader)
    """

    def reader():
        rs = []
        for r in readers:
            rs.append(r())
        yield from map(func, *rs)

    return reader


def shuffle(reader: _Reader[_T], buf_size: int) -> _Reader[_T]:
    """
    This API creates a decorated reader that outputs the shuffled data.

    The output data from the origin reader will be saved into a buffer,
    and then shuffle the data. The size of buffer is determined by argument buf_size.

    Args:
        reader(callable): the original reader whose data will be shuffled.
        buf_size(int): the size of shuffled buffer.

    Returns:
        callable: a decorated reader.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('outputs are 0~4 unordered arrangement')
            >>> def reader():
            ...     for i in range(5):
            ...         yield i
            >>> shuffled_reader = paddle.reader.decorator.shuffle(reader, 3)
            >>> for e in shuffled_reader():
            ...     print(e)
            >>> # outputs are 0~4 unordered arrangement
    """

    def data_reader() -> Generator[_T, None, None]:
        buf = []
        for e in reader():
            buf.append(e)
            if len(buf) >= buf_size:
                random.shuffle(buf)
                for b in buf:
                    yield b
                buf = []

        if len(buf) > 0:
            random.shuffle(buf)
            for b in buf:
                yield b

    return data_reader


def chain(*readers: _Reader[_T]) -> _Reader[_T]:
    """
    Use the input data readers to create a chained data reader. The new created reader
    chains the outputs of input readers together as its output, and it do not change
    the format of the outputs.

    **Note**:
        ``paddle.reader.chain`` is the alias of ``paddle.base.io.chain``, and
        ``paddle.base.io.chain`` is recommended to use.

    For example, if three input readers' outputs are as follows:
    [0, 0, 0],
    [10, 10, 10],
    [20, 20, 20].
    The chained reader will output:
    [0, 0, 0], [10, 10, 10], [20, 20, 20].

    Args:
        readers(list): input data readers.

    Returns:
        callable: the new chained data reader.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> def reader_creator_3(start):
            ...     def reader():
            ...         for i in range(start, start + 3):
            ...             yield [i, i, i]
            ...     return reader
            ...
            >>> c = paddle.reader.chain(reader_creator_3(0), reader_creator_3(10), reader_creator_3(20))
            >>> for e in c():
            ...     print(e)
            [0, 0, 0]
            [1, 1, 1]
            [2, 2, 2]
            [10, 10, 10]
            [11, 11, 11]
            [12, 12, 12]
            [20, 20, 20]
            [21, 21, 21]
            [22, 22, 22]

    """

    def reader() -> Generator[_T, None, None]:
        rs: list[Generator[_T, None, None]] = []
        for r in readers:
            rs.append(r())

        yield from itertools.chain(*rs)

    return reader


class ComposeNotAligned(ValueError):
    pass


def compose(
    *readers: _Reader[Any], **kwargs: Unpack[_ComposeOptions]
) -> _Reader[Any]:
    """
    Creates a data reader whose output is the combination of input readers.

    If input readers output following data entries:
    (1, 2)    3    (4, 5)
    The composed reader will output:
    (1, 2, 3, 4, 5)

    Args:
        readers (Reader|list of Reader): readers that will be composed together.
        check_alignment(bool, optional): Indicates whether the input readers are checked for
                              alignment. If True, whether input readers are aligned
                              correctly will be checked, else alignment will not be checkout and trailing outputs
                              will be discarded. Defaults to True.

    Returns:
        the new data reader (Reader).

    Examples:
        .. code-block:: python

            >>> def reader_creator_10(dur):
            ...     def reader():
            ...         for i in range(10):
            ...             yield i
            ...     return reader
            >>> reader = paddle.reader.decorator.compose(reader_creator_10(0), reader_creator_10(0))
    """
    check_alignment = kwargs.pop('check_alignment', True)

    def make_tuple(x):
        if isinstance(x, tuple):
            return x
        else:
            return (x,)

    def reader():
        rs = []
        for r in readers:
            rs.append(r())
        if not check_alignment:
            for outputs in zip(*rs):
                yield sum(list(map(make_tuple, outputs)), ())
        else:
            for outputs in zip_longest(*rs):
                for o in outputs:
                    if o is None:
                        # None will be not be present if compose is aligned
                        raise ComposeNotAligned(
                            "outputs of readers are not aligned."
                        )
                yield sum(list(map(make_tuple, outputs)), ())

    return reader


def buffered(reader: _Reader[_T], size: int) -> _Reader[_T]:
    """
    Creates a buffered data reader.

    The buffered data reader will read and save data entries into a
    buffer. Reading from the buffered data reader will proceed as long
    as the buffer is not empty.

    Args:
        reader(generator): the data reader to read from.
        size(int): max buffer size.

    Returns:
        generator: the buffered data reader.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> def reader():
            ...     for i in range(3):
            ...         yield i
            ...
            >>> # Create a buffered reader, and the buffer size is 2.
            >>> buffered_reader = paddle.reader.decorator.buffered(reader, 2)

            >>> # Output: 0 1 2
            >>> for i in buffered_reader():
            ...     print(i)
            0
            1
            2
    """

    class EndSignal:
        pass

    end = EndSignal()

    def read_worker(r, q):
        for d in r:
            q.put(d)
        q.put(end)

    def data_reader():
        r = reader()
        q = Queue(maxsize=size)
        t = Thread(
            target=read_worker,
            args=(r, q),
        )
        t.daemon = True
        t.start()
        e = q.get()
        while e != end:
            yield e
            e = q.get()

    return data_reader


def firstn(reader: _Reader[_T], n: int) -> _Reader[_T]:
    """

    This API creates a decorated reader, and limits the max number of
    samples that reader could return.

    Args:
        reader(callable): the input reader.
        n(int): the max number of samples in the reader.

    Returns:
        callable: the decorated reader.

    Examples:
        .. code-block:: python

            >>> def reader():
            ...     for i in range(100):
            ...         yield i
            >>> firstn_reader = paddle.reader.decorator.firstn(reader, 5)
            >>> for e in firstn_reader():
            ...     print(e)
            0
            1
            2
            3
            4
    """

    # TODO(yuyang18): Check if just drop the reader, could clean the opened
    # resource or not?

    def firstn_reader():
        for i, item in enumerate(reader()):
            if i == n:
                break
            yield item

    return firstn_reader


class XmapEndSignal:
    pass


def xmap_readers(
    mapper: Callable[[_T], _U],
    reader: _Reader[_T],
    process_num: int,
    buffer_size: int,
    order: bool = False,
) -> _Reader[_U]:
    """
    Use multi-threads to map samples from reader by a mapper defined by user.

    Args:
        mapper (callable): a function to map the data from reader.
        reader (callable): a data reader which yields the data.
        process_num (int): thread number to handle original sample.
        buffer_size (int): size of the queue to read data in.
        order (bool): whether to keep the data order from original reader.
            Default False.

    Returns:
        callable: a decorated reader with data mapping.
    """
    end = XmapEndSignal()

    # define a worker to read samples from reader to in_queue
    def read_worker(reader, in_queue):
        for i in reader():
            in_queue.put(i)
        in_queue.put(end)

    # define a worker to read samples from reader to in_queue with order flag
    def order_read_worker(reader, in_queue):
        in_order = 0
        for i in reader():
            in_queue.put((in_order, i))
            in_order += 1
        in_queue.put(end)

    # define a worker to handle samples from in_queue by mapper
    # and put mapped samples into out_queue
    def handle_worker(in_queue, out_queue, mapper):
        sample = in_queue.get()
        while not isinstance(sample, XmapEndSignal):
            r = mapper(sample)
            out_queue.put(r)
            sample = in_queue.get()
        in_queue.put(end)
        out_queue.put(end)

    # define a worker to handle samples from in_queue by mapper
    # and put mapped samples into out_queue by order
    def order_handle_worker(in_queue, out_queue, mapper, out_order):
        ins = in_queue.get()
        while not isinstance(ins, XmapEndSignal):
            order, sample = ins
            r = mapper(sample)
            while order != out_order[0]:
                pass
            out_queue.put(r)
            out_order[0] += 1
            ins = in_queue.get()
        in_queue.put(end)
        out_queue.put(end)

    def xreader():
        in_queue = Queue(buffer_size)
        out_queue = Queue(buffer_size)
        out_order = [0]
        # start a read worker in a thread
        target = order_read_worker if order else read_worker
        t = Thread(target=target, args=(reader, in_queue))
        t.daemon = True
        t.start()
        # start several handle_workers
        target = order_handle_worker if order else handle_worker
        args = (
            (in_queue, out_queue, mapper, out_order)
            if order
            else (in_queue, out_queue, mapper)
        )
        workers = []
        for i in range(process_num):
            worker = Thread(target=target, args=args)
            worker.daemon = True
            workers.append(worker)
        for w in workers:
            w.start()

        sample = out_queue.get()
        while not isinstance(sample, XmapEndSignal):
            yield sample
            sample = out_queue.get()
        finish = 1
        while finish < process_num:
            sample = out_queue.get()
            if isinstance(sample, XmapEndSignal):
                finish += 1
            else:
                yield sample

    return xreader


def multiprocess_reader(
    readers: Sequence[_Reader[_T]],
    use_pipe: bool = True,
    queue_size: int = 1000,
) -> _Reader[list[_T]]:
    """
    This API use python ``multiprocessing`` to read data from ``readers`` parallelly,
    and then ``multiprocess.Queue`` or ``multiprocess.Pipe`` is used to merge
    these data. A separate process will be created for each reader in the
    ``readers`` list, please guarantee every reader can work independently
    to avoid conflicts in parallel environment.


    ``Multiprocess.Queue`` require the rw access right to /dev/shm, and it's not supported
    in some platforms.

    Parameters:
        readers (list( ``generator`` ) | tuple( ``generator`` )): a python ``generator`` list
            used to read input data
        use_pipe (bool, optional): control the inner API used to implement the multi-processing,
            default True - use ``multiprocess.Pipe`` which is recommended
        queue_size (int, optional): only useful when ``use_pipe`` is False - ``multiprocess.Queue``
            is used, default 1000. Increase this value can speed up the data reading, and more memory
            will be consumed.

    Returns:
        ``generator``: a new reader which can be run parallelly


    Example:

        .. code-block:: python

            >>> import paddle
            >>> import numpy as np

            >>> sample_files = ['sample_file_1', 'sample_file_2']

            >>> def fake_input_files():
            ...     with open(sample_files[0], 'wb') as f:
            ...         np.savez(f, a=np.array([1, 2]), b=np.array([3, 4]), c=np.array([5, 6]), d=np.array([7, 8]))
            ...     with open(sample_files[1], 'wb') as f:
            ...         np.savez(f, a=np.array([9, 10]), b=np.array([11, 12]), c=np.array([13, 14]))
            ...
            ...
            >>> def generate_reader(file_name):
            ...     # load data file
            ...     def _impl():
            ...         data = np.load(file_name)
            ...         for item in sorted(data.files):
            ...             yield data[item],
            ...     return _impl
            ...
            >>> if __name__ == '__main__':
            ...     # generate sample input files
            ...     fake_input_files()
            ...
            ...     with base.program_guard(base.Program(), base.Program()):
            ...         place = base.CPUPlace()
            ...         # the 1st 2 is batch size
            ...
            ...         image = paddle.static.data(name='image', dtype='int64', shape=[2, 1, 2])
            ...         paddle.static.Print(image)
            ...         # print detailed tensor info of image variable
            ...
            ...         reader = base.io.PyReader(feed_list=[image], capacity=2)
            ...
            ...         decorated_reader = paddle.reader.multiprocess_reader(
            ...             [generate_reader(sample_files[0]), generate_reader(sample_files[1])], False)
            ...
            ...         reader.decorate_sample_generator(decorated_reader, batch_size=2, places=[place])
            ...
            ...         exe = base.Executor(place)
            ...         exe.run(base.default_startup_program())
            ...
            ...         for data in reader():
            ...             res = exe.run(feed=data, fetch_list=[image])
            ...             print(res[0])
            [[[1 2]], [[3 4]]]
            [[[5 6]], [[7 8]]]
            [[[9 10]], [[11 12]]]
    """

    if sys.platform == 'win32':
        raise NotImplementedError(
            "The multiprocess_reader method is not supported on windows."
        )

    # ujson is ultra fast json encoder and decoder written in pure C with bindings for Python 3.6+.
    try:
        import ujson as json
    except Exception as e:
        warnings.warn(
            "The `ujson` module is not found, use the `json` module, `ujson` encodes and decodes faster, "
            "you can install `ujson` through `pip install ujson`."
        )
        import json

    assert (
        isinstance(readers, (list, tuple)) and len(readers) > 0
    ), "`readers` must be list or tuple."

    def _read_into_queue(reader, queue):
        try:
            for sample in reader():
                if sample is None:
                    raise ValueError("sample has None")
                queue.put(sample)
            queue.put(None)
        except Exception as e:
            queue.put("")
            raise e

    def queue_reader():
        queue = fork_context.Queue(queue_size)
        for reader in readers:
            p = fork_context.Process(
                target=_read_into_queue, args=(reader, queue)
            )
            p.start()

        reader_num = len(readers)
        finish_num = 0
        while finish_num < reader_num:
            try:
                sample = queue.get(timeout=QUEUE_GET_TIMEOUT)
            except Exception as e:
                logging.error(
                    "multiprocess_reader failed to get data from the multiprocessing.Queue."
                )
                raise e

            if sample is None:
                finish_num += 1
            elif sample == "":
                raise ValueError(
                    "multiprocess_reader failed to put data into the multiprocessing.Queue."
                )
            else:
                yield sample

    def _read_into_pipe(reader, conn):
        try:
            for sample in reader():
                if sample is None:
                    raise ValueError("sample has None!")
                conn.send(json.dumps(sample))
            conn.send(json.dumps(None))
            conn.close()
        except Exception as e:
            conn.send(json.dumps(""))
            conn.close()
            raise e

    def pipe_reader():
        conns = []
        for reader in readers:
            parent_conn, child_conn = fork_context.Pipe()
            conns.append(parent_conn)
            p = fork_context.Process(
                target=_read_into_pipe, args=(reader, child_conn)
            )
            p.start()

        reader_num = len(readers)
        finish_num = 0
        conn_to_remove = []
        while finish_num < reader_num:
            for conn in conn_to_remove:
                conns.remove(conn)
            conn_to_remove = []
            for conn in conns:
                sample = json.loads(conn.recv())
                if sample is None:
                    finish_num += 1
                    conn.close()
                    conn_to_remove.append(conn)
                elif sample == "":
                    conn.close()
                    conn_to_remove.append(conn)
                    raise ValueError(
                        "multiprocess_reader failed to send data into the multiprocessing.Pipe."
                    )
                else:
                    yield sample

    if use_pipe:
        return pipe_reader
    else:
        return queue_reader
