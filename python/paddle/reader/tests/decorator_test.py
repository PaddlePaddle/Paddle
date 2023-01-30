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

<<<<<<< HEAD
import functools
import sys
import time
import unittest
=======
import sys
import time
import unittest
import functools
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

import paddle.reader

__all__ = []


def reader_creator_10(dur):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def reader():
        for i in range(10):
            # this invocation helps testing paddle.reader.buffer
            time.sleep(dur)
            yield i

    return reader


class TestMap(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_map(self):
        d = {"h": 0, "i": 1}

        def tokenize(x):
            return d[x]

        def read():
            yield "h"
            yield "i"

        r = paddle.reader.map_readers(tokenize, read)
        for i, e in enumerate(r()):
            self.assertEqual(e, i)


class TestBuffered(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_read(self):
        for size in range(20):
            b = paddle.reader.buffered(reader_creator_10(0), size)
            c = 0
            for i in b():
                self.assertEqual(i, c)
                c += 1
            self.assertEqual(c, 10)

    def test_buffering(self):
        # read have 30ms delay.
        b = paddle.reader.buffered(reader_creator_10(0.03), 10)
        last_time = time.time()
        for idx, i in enumerate(b()):
            elapsed_time = time.time() - last_time
            if i == 0:
                time.sleep(1)
            else:
                # read time should be short, meaning already buffered.
                self.assertLess(elapsed_time, 0.08)
            last_time = time.time()


class TestCompose(unittest.TestCase):
<<<<<<< HEAD
    def test_compse(self):
        reader = paddle.reader.compose(
            reader_creator_10(0), reader_creator_10(0)
        )
=======

    def test_compse(self):
        reader = paddle.reader.compose(reader_creator_10(0),
                                       reader_creator_10(0))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        for idx, e in enumerate(reader()):
            self.assertEqual(e, (idx, idx))

    def test_compose_not_aligned(self):
        total = 0
        reader = paddle.reader.compose(
            paddle.reader.chain(reader_creator_10(0), reader_creator_10(0)),
<<<<<<< HEAD
            reader_creator_10(0),
        )
=======
            reader_creator_10(0))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with self.assertRaises(paddle.reader.ComposeNotAligned):
            for e in reader():
                total += 1
        # expecting 10, not 20
        self.assertEqual(total, 10)

    def test_compose_not_aligned_no_check(self):
        total = 0
<<<<<<< HEAD
        reader = paddle.reader.compose(
            paddle.reader.chain(reader_creator_10(0), reader_creator_10(0)),
            reader_creator_10(0),
            check_alignment=False,
        )
=======
        reader = paddle.reader.compose(paddle.reader.chain(
            reader_creator_10(0), reader_creator_10(0)),
                                       reader_creator_10(0),
                                       check_alignment=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        for e in reader():
            total += 1
        # expecting 10, not 20
        self.assertEqual(total, 10)


class TestChain(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_chain(self):
        c = paddle.reader.chain(reader_creator_10(0), reader_creator_10(0))
        idx = 0
        for e in c():
            self.assertEqual(e, idx % 10)
            idx += 1
        self.assertEqual(idx, 20)


class TestShuffle(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_shuffle(self):
        case = [(0, True), (1, True), (10, False), (100, False)]
        a = reader_creator_10(0)
        for size, checkEq in case:
            s = paddle.reader.shuffle(a, size)
            total = 0
            for idx, e in enumerate(s()):
                if checkEq:
                    self.assertEqual(idx, e)
                total += 1
            self.assertEqual(total, 10)


class TestXmap(unittest.TestCase):
<<<<<<< HEAD
    def test_xmap(self):
        def mapper(x):
            return x + 1
=======

    def test_xmap(self):

        def mapper(x):
            return (x + 1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        orders = (True, False)
        thread_nums = (1, 2, 4, 8, 16)
        buffered_size = (1, 2, 4, 8, 16)
        for order in orders:
            for tNum in thread_nums:
                for size in buffered_size:
<<<<<<< HEAD
                    reader = paddle.reader.xmap_readers(
                        mapper, reader_creator_10(0), tNum, size, order
                    )
=======
                    reader = paddle.reader.xmap_readers(mapper,
                                                        reader_creator_10(0),
                                                        tNum, size, order)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    for n in range(3):
                        result = []
                        for i in reader():
                            result.append(i)
                        if not order:
                            result.sort()
                        for idx, e in enumerate(result):
                            self.assertEqual(e, mapper(idx))


class TestMultiProcessReader(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setup(self):
        self.samples = []
        for i in range(1000):
            self.samples.append([[i], [i + 1, i + 2], i + 3])

        def reader(index):
            for i in range(len(self.samples)):
                if i % 3 == index:
                    yield self.samples[i]

        self.reader0 = functools.partial(reader, 0)
        self.reader1 = functools.partial(reader, 1)
        self.reader2 = functools.partial(reader, 2)

    def reader_test(self, use_pipe):
        self.setup()
        results = []
        for data in paddle.reader.multiprocess_reader(
<<<<<<< HEAD
            [self.reader0, self.reader1, self.reader2], 100, use_pipe
        )():
=======
            [self.reader0, self.reader1, self.reader2], 100, use_pipe)():
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            results.append(data)
        self.assertEqual(sorted(self.samples), sorted(results))

    def test_distributed_batch_reader(self):
        if sys.platform != 'win32':
            self.reader_test(use_pipe=False)
            self.reader_test(use_pipe=True)


if __name__ == '__main__':
    unittest.main()
