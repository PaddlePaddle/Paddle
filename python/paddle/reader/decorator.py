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


def shuffle(reader_creator, buf_size):
    """Creates a data reader creator whose data output is suffled.

    Output from the iterator that created by original reader creator will be
    buffered into shuffle buffer, and then shuffled. The size of shuffle buffer
    is determined by argument buf_size.

    Args:
        reader_creator: the original reader creator whose output will be
            shuffled.
        buf_size: shuffle buffer size.

    Returns:
        the new reader creator whose output is shuffled.
    """

    def create_reader_creator():
        buf = []
        for e in reader_creator():
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

    return create_reader_creator


def chain(*reader_creators):
    """Creates a data reader creator whose output is the outputs of input data
       reader creators chained together.

    If input reader creators output following data entries:
    [0, 0, 0]
    [1, 1, 1]
    [2, 2, 2]
    The chained reader creator will output:
    [0, 0, 0, 1, 1, 1, 2, 2, 2]

    Args:
        readers_creators: input reader creators

    Returns:
        the new data reader creator.
    """

    def create_reader_creator():
        rs = []
        for r in reader_creators:
            rs.append(r())

        for e in itertools.chain(*rs):
            yield e

    return create_reader_creator


class ComposeNotAligned:
    pass


def compose(*reader_creators, **kwargs):
    """Creates a data reader creator whose output is the combination of input
       readers creators.

    If input reader creators output following data entries:
    (1, 2)    3    (4, 5)
    The composed reader creator will output:
    (1, 2, 3, 4, 5)

    Args:
        *reader_creators: reader creators that will be composed together.
        check_alignment: If True, will check if input reader creators are aligned
            correctly. If False, will not check alignment and trailing outputs
            will be discarded. Defaults to True.

    Returns:
        the new data reader creator.

    Raises:
        ComposeNotAligned: outputs of reader creators are not aligned.
            Will not raise when check_alignment is set to False.
    """
    check_alignment = kwargs.pop('check_alignment', True)

    def make_tuple(x):
        if isinstance(x, tuple):
            return x
        else:
            return (x, )

    def create_reader_creator():
        rs = []
        for r in reader_creators:
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

    return create_reader_creator


def buffered(reader_creator, size):
    """Creates a buffered data reader creator.

    The buffered data reader creator will read and save data entries into a
    buffer. Reading from the buffered data reader creator will proceed as long
    as the buffer is not empty.
    
    Args:
        reader_creator: the data reader creator to read from.
        size: max buffer size.
    
    Returns:
        The buffered data reader creator.
    """

    class EndSignal():
        pass

    end = EndSignal()

    def read_worker(r, q):
        for d in r:
            q.put(d)
        q.put(end)

    def create_reader_creator():
        r = reader_creator()
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

    return create_reader_creator
