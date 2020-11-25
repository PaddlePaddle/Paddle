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

from __future__ import print_function

import unittest

import paddle
import paddle.fluid as fluid

from test_desc_clone import get_model, program_equal


def get_transpiler(trainer_id, main_program, pserver_endpoints, trainers):
    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=trainer_id,
        program=main_program,
        pservers=pserver_endpoints,
        trainers=trainers)
    return t


class TestDistMnist(unittest.TestCase):
    def test_desc_clone(self):
        paddle.enable_static()
        get_model(batch_size=20)

        pserver_endpoints = "127.0.0.1:9123"
        trainers = 1
        current_endpoint = "127.0.0.1:9123"
        t = get_transpiler(0,
                           fluid.default_main_program(), pserver_endpoints,
                           trainers)

        pserver_prog = t.get_pserver_program(current_endpoint)
        startup_prog = t.get_startup_program(current_endpoint, pserver_prog)
        main = pserver_prog.clone()
        startup = startup_prog.clone()
        self.assertTrue(program_equal(main, pserver_prog))
        self.assertTrue(program_equal(startup, startup_prog))
