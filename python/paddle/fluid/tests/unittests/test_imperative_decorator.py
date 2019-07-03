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

import paddle.fluid as fluid
import paddle.fluid.framework as framework
import unittest


class TestTracerMode(unittest.TestCase):
    def setUp(self):
        self.init_mode = True

    def get_tracer_mode(self):
        assert fluid.dygraph.enabled(), "Dygraph mode must be enabled"

    @fluid.dygraph.no_grad
    def no_grad_func(self, a):
        self.assertEqual(self.tracer._train_mode, False)
        return a

    def test_main(self):
        with fluid.dygraph.guard():
            self.tracer = framework._dygraph_tracer()
            self.tracer._train_mode = self.init_mode

            self.assertEqual(self.no_grad_func(1), 1)

            self.assertEqual(self.tracer._train_mode, self.init_mode)


class TestTracerMode2(TestTracerMode):
    def setUp(self):
        self.init_mode = False


if __name__ == '__main__':
    unittest.main()
