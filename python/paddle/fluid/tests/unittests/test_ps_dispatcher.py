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

import unittest
from paddle.fluid.incubate.fleet.parameter_server.ir.ps_dispatcher import RoundRobin, HashName, PSDispatcher


class TestPsDispatcher(unittest.TestCase):

    def setUp(self):
        self.points = [
            "127.0.0.1:1001", "127.0.0.1:1002", "127.0.0.1:1003",
            "127.0.0.1:1004"
        ]

    def test_base(self):
        base = PSDispatcher(self.points)
        self.assertEqual(len(base.eps), 4)
        base.reset()

        with self.assertRaises(NotImplementedError):
            base.dispatch([])

    def test_hash(self):

        class Var:

            def __init__(self, index):
                self._name = "var_{}".format(index)

            def name(self):
                return self._name

        xx = HashName(self.points)
        self.assertEqual(len(xx.eps), 4)
        xx.reset()

        vars = []
        for i in range(4):
            vars.append(Var(i))
        eplist = xx.dispatch(vars)
        self.assertEqual(len(eplist), 4)

    def test_round_rodin(self):

        class Var:

            def __init__(self, index):
                self._name = "var_{}".format(index)

            def name(self):
                return self._name

        xx = RoundRobin(self.points)
        self.assertEqual(len(xx.eps), 4)
        xx.reset()

        vars = []
        for i in range(4):
            vars.append(Var(i))
        eplist = xx.dispatch(vars)
        self.assertEqual(len(eplist), 4)


if __name__ == '__main__':
    unittest.main()
