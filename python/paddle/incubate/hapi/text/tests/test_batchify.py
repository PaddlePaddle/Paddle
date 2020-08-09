#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.inbubate.hapi.text import Stack, Pad, Tuple, BertTokenizer
from paddle.inbubate.hapi.text import BertForSequenceClassification
import unittest


class TestBatchify(unittest.TestCase):
    def setUp(self):
    	self.a = [1, 2, 3, 4]
    	self.b = [5, 6, 7, 8]
    	self.c = [9, 1, 2, 3]
    	self.d = [1, 2]
    	self.e = [3, 5, 6]

    def testCall(self):
    	data = self.a + self.b + self.c
    	stack_data = Stack(data)
    	assert stack_data.shape == [3, 4]

    def testPad(self):
    	pad_data = Pad(pad_val=0)([a, d, e])
    	assert pad_data.shape == [3, 4]

	def testTuple(self):
        tuple_fn = Tuple(Stack, Pad)
        tuple_fn(self.a, self.b, self.c)
