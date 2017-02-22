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

__all__ = ['buffered', 'compose', 'chain', 'shuffle', 'ComposeNotAligned']

from Queue import Queue
from threading import Thread
import itertools
import random


def shuffle(reader, buf_size):
    """Creates a data reader whose data output is suffled.

    Output from the iterator that created by original reader will be
    buffered into shuffle buffer, and then shuffled. The size of shuffle buffer
    is determined by argument buf_size.

    Args:
        reader: the original reader whose output will be
            shuffled.
        buf_size: shuffle buffer size.

    Returns:
        the new reader whose output is shuffled.
    """

    def data_reader():
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


def chain(*readers):
    """Creates a data reader whose output is the outputs of input data
       readers chained together.

    If input readers output following data entries:
    [0, 0, 0]
    [1, 1, 1]
    [2, 2, 2]
    The chained reader will output:
    [0, 0, 0, 1, 1, 1, 2, 2, 2]

    Args:
        readerss: input readers.

    Returns:
        the new data reader.
    """

    def reader():
        rs = []
        for r in readers:
            rs.append(r())

        for e in itertools.chain(*rs):
            yield e

    return reader


class ComposeNotAligned:
    pass


def compose(*readers, **kwargs):
    """Creates a data reader whose output is the combination of input readers.

    If input readers output following data entries:
    (1, 2)    3    (4, 5)
    The composed reader will output:
    (1, 2, 3, 4, 5)

    Args:
        *readers: readers that will be composed together.
        check_alignment: If True, will check if input readers are aligned
            correctly. If False, will not check alignment and trailing outputs
            will be discarded. Defaults to True.

    Returns:
        the new data reader.

    Raises:
        ComposeNotAligned: outputs of readers are not aligned.
            Will not raise when check_alignment is set to False.
    """
    check_alignment = kwargs.pop('check_alignment', True)

    def make_tuple(x):
        if isinstance(x, tuple):
            return x
        else:
            return (x, )

    def reader():
        rs = []
        for r in readers:
            rs.append(r())
        if not check_alignment:
            for outputs in itertools.izip(*rs):
                yield sum(map(make_tuple, outputs), ())
        else:
            for outputs in itertools.izip_longest(*rs):
                for o in outputs:
                    if o is None:
                        # None will be not be present if compose is aligned
                        raise ComposeNotAligned
                yield sum(map(make_tuple, outputs), ())

    return reader


def buffered(reader, size):
    """Creates a buffered data reader.

    The buffered data reader will read and save data entries into a
    buffer. Reading from the buffered data reader will proceed as long
    as the buffer is not empty.
    
    Args:
        reader: the data reader to read from.
        size: max buffer size.
    
    Returns:
        The buffered data reader.
    """

    class EndSignal():
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
            target=read_worker, args=(
                r,
                q, ))
        t.daemon = True
        t.start()
        e = q.get()
        while e != end:
            yield e
            e = q.get()

    return data_reader
