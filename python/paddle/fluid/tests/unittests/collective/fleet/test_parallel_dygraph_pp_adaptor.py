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

import shutil
import unittest

from test_parallel_dygraph_dataparallel import TestMultipleGpus

import paddle
from paddle.fleet.utils.pp_parallel_adaptor import (
    ParallelConfig,
    PipeLineModelAdaptor,
)


class TestHybridPipeParallel(TestMultipleGpus):
    def test_hybrid_parallel_transformer_unbalanced_data(self):
        self.run_mnist_2gpu('hybrid_parallel_pp_transformer_save.py')
        self.run_mnist_2gpu(
            'hybrid_parallel_pp_transformer_save_with_virtual_stage.py'
        )
        # test pp adaptor
        dir1 = "./pp_transformer"
        p_config1 = ParallelConfig(mp=1, pp=2, vpp=1, sharding=1)
        dir2 = "./pp_transformer_vp"
        p_config2 = ParallelConfig(mp=1, pp=2, vpp=2, sharding=1)

        pp_to_vp = PipeLineModelAdaptor(p_config1, p_config2)
        vp_to_pp = PipeLineModelAdaptor(p_config2, p_config1)

        def check_params_names(dir1, dir2):
            for i in p_config1.pp:
                params_1 = paddle.load(
                    "{}/mp_00_sharding_00_pp_{:0>2d}/model.pdparams".format(
                        dir1, i
                    )
                )
                params_2 = paddle.load(
                    "{}/mp_00_sharding_00_pp_{:0>2d}/model.pdparams".format(
                        dir2, i
                    )
                )
                self.assertEqual(params_1.keys(), params_2.keys())

        # check pp to vp
        tmp_dir1 = "./tmp_vp"
        pp_to_vp.apply(dir1, tmp_dir1)
        check_params_names(tmp_dir1, dir2)

        # check vp to pp
        tmp_dir2 = "./tmp_pp"
        vp_to_pp.apply(dir2, tmp_dir2)
        check_params_names(tmp_dir2, dir1)

        # rm dirs
        shutil.rmtree(dir1, ignore_errors=True)
        shutil.rmtree(dir2, ignore_errors=True)
        shutil.rmtree(tmp_dir1, ignore_errors=True)
        shutil.rmtree(tmp_dir2, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
