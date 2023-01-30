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
<<<<<<< HEAD

=======
import paddle
import itertools
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import test_communication_api_base as test_base


class TestCommunicationStreamReduceAPI(test_base.CommunicationTestDistBase):
<<<<<<< HEAD
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=120)
=======

    def setUp(self):
        super(TestCommunicationStreamReduceAPI, self).setUp(num_of_devices=2,
                                                            timeout=120)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._default_envs = {
            "backend": "nccl",
            "shape": "(100, 200)",
            "dtype": "float32",
<<<<<<< HEAD
            "seeds": str(self._seeds),
        }
        self._changeable_envs = {
            "sync_op": ["True", "False"],
            "use_calc_stream": ["True", "False"],
        }

    def test_reduce_stream(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            if eval(envs["use_calc_stream"]) and not eval(envs["sync_op"]):
                continue
            self.run_test_case(
                "communication_stream_reduce_api_dygraph.py",
                user_defined_envs=envs,
            )

    def tearDown(self):
        super().tearDown()
=======
            "seeds": str(self._seeds)
        }
        self._changeable_envs = {
            "sync_op": ["True", "False"],
            "use_calc_stream": ["True", "False"]
        }

    def test_reduce_stream(self):
        envs_list = test_base.gen_product_envs_list(self._default_envs,
                                                    self._changeable_envs)
        for envs in envs_list:
            if eval(envs["use_calc_stream"]) and not eval(envs["sync_op"]):
                continue
            self.run_test_case("communication_stream_reduce_api_dygraph.py",
                               user_defined_envs=envs)

    def tearDown(self):
        super(TestCommunicationStreamReduceAPI, self).tearDown()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
