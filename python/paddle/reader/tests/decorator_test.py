# Copyright PaddlePaddle contributors. All Rights Reserved
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
import unittest
import paddle.reader
import time


def reader_10(dur):
    for i in range(10):
        time.sleep(dur)
        yield i


class TestBuffered(unittest.TestCase):
    def test_read(self):
        for size in range(20):
            b = paddle.reader.buffered(lambda: reader_10(0), size)
            c = 0
            for i in b():
                self.assertEqual(i, c)
                c += 1
            self.assertEqual(c, 10)

    def test_buffering(self):
        # read have 30ms delay.
        b = paddle.reader.buffered(lambda: reader_10(0.03), 10)
        last_time = time.time()
        for idx, i in enumerate(b()):
            elapsed_time = time.time() - last_time
            if i == 0:
                time.sleep(0.3)
            else:
                # read time should be short, meaning already buffered.
                self.assertLess(elapsed_time, 0.01)
            last_time = time.time()


if __name__ == '__main__':
    unittest.main()
