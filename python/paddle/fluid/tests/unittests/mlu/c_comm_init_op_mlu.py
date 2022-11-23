# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.distributed.fleet.base.private_helper_function import wait_server_ready
import paddle

paddle.enable_static()


class TestCCommInitOp(unittest.TestCase):

    def setUp(self):
        self.endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS").split(',')
        self.current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        self.nranks = len(self.endpoints)
        self.rank = self.endpoints.index(self.current_endpoint)
        self.mlu_id = int(os.getenv("FLAGS_selected_mlus"))
        self.place = fluid.MLUPlace(self.mlu_id)
        self.exe = fluid.Executor(self.place)
        self.endpoints.remove(self.current_endpoint)
        self.other_endpoints = self.endpoints
        if self.rank == 0:
            wait_server_ready(self.other_endpoints)

    def test_specifying_devices(self):
        program = fluid.Program()
        block = program.global_block()
        cncl_id_var = block.create_var(
            name=fluid.unique_name.generate('cncl_id'),
            persistable=True,
            type=fluid.core.VarDesc.VarType.RAW)
        block.append_op(type='c_gen_cncl_id',
                        inputs={},
                        outputs={'Out': cncl_id_var},
                        attrs={
                            'rank': self.rank,
                            'endpoint': self.current_endpoint,
                            'other_endpoints': self.other_endpoints
                        })
        block.append_op(type='c_comm_init',
                        inputs={'X': cncl_id_var},
                        outputs={},
                        attrs={
                            'nranks': self.nranks,
                            'rank': self.rank,
                            'ring_id': 0,
                            'device_id': self.mlu_id
                        })
        self.exe.run(program)


if __name__ == "__main__":
    unittest.main()
