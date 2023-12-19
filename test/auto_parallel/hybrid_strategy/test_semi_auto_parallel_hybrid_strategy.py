# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import tempfile
import unittest

import collective.test_communication_api_base as test_base

import paddle


class TestSemiAutoParallelDPMPStrategy(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=4, timeout=120, nnode=1)
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
        }
        self._changeable_envs = {"backend": ["gpu"]}

    def test_simple_net_hybrid_strategy(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "semi_auto_parallel_simple_net_dp_mp.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()


class TestSemiAutoParallelHybridStrategy(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=8,
            timeout=120,
            nnode=1,
        )
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
        }
        self._changeable_envs = {"backend": ["gpu"]}

    def test_simple_net_hybrid_strategy(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "semi_auto_parallel_simple_net_dp_mp_pp.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()


class TestSemiAutoParallelHybridStrategyWithSP(
    test_base.CommunicationTestDistBase
):
    def setUp(self):
        super().setUp(
            num_of_devices=4,
            timeout=120,
            nnode=1,
        )
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
        }
        self._changeable_envs = {"backend": ["gpu"], "is_dp": ["false"]}

    def test_simple_net_mp_pp_sp(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "semi_auto_parallel_simple_net_sp.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()

    def test_simple_net_dp_mp_pp_sp(self):
        super().setUp(
            num_of_devices=8,
            timeout=120,
            nnode=1,
        )
        self._changeable_envs = {"backend": ["gpu"], "is_dp": ["true"]}
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "semi_auto_parallel_simple_net_sp.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()


class TestSemiAutoParallelCrossMeshReshard(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=4,
            timeout=120,
            nnode=1,
        )
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
        }
        self._changeable_envs = {"backend": ["gpu"]}

    def test_simple_net_cross_mesh_reshard(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_cross_mesh_reshard.py",
                user_defined_envs=envs,
            )


class TestSemiAutoParallelNdCrossMeshReshard(
    test_base.CommunicationTestDistBase
):
    def setUp(self):
        super().setUp(num_of_devices=8, timeout=200, nnode=1)
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
        }
        self._changeable_envs = {"backend": ["gpu"]}

    def test_simple_net_bybrid_strategy(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_nd_cross_mesh_reshard.py",
                user_defined_envs=envs,
            )


class TestSemiAutoParallelLlamaDPMPStrategy(
    test_base.CommunicationTestDistBase
):
    def setUp(self):
        super().setUp(num_of_devices=4, timeout=200, nnode=1)
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
        }
        self._changeable_envs = {"backend": ["gpu"]}

    def test_simple_net_hybrid_strategy(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        cuda_version_main = int(paddle.version.cuda().split(".")[0])
        device_prop_main = paddle.device.cuda.get_device_capability()[0]
        if cuda_version_main >= 11 and device_prop_main >= 8:
            for envs in envs_list:
                self.run_test_case(
                    "semi_auto_parallel_for_llama_decoder_dp_mp.py",
                    user_defined_envs=envs,
                )


class TestSemiAutoParallelLlama2D(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=4, timeout=200, nnode=1)
        self._default_envs = {"dp": "2", "mp": "2", "pp": "1", "acc_step": "2"}
        self._changeable_envs = {
            "backend": ["gpu"],
            "use_sp": ["true", "false"],
        }

    def test_simple_net_hybrid_strategy(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_llama.py",
                user_defined_envs=envs,
            )


class TestSemiAutoParallelLlama3D(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=8, timeout=200, nnode=1)
        self._default_envs = {"dp": "2", "mp": "2", "pp": "2", "acc_step": "2"}
        self._changeable_envs = {
            "backend": ["gpu"],
            "use_sp": ["true", "false"],
        }

    def test_simple_net_hybrid_strategy(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_llama.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()
