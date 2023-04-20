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

import os
import shutil
import unittest

from test_parallel_dygraph_dataparallel import TestMultipleGpus

import paddle
from paddle.distributed.fleet.utils.pp_parallel_adaptor import (
    ParallelConfig,
    PipeLineModelAdaptor,
)


class TestHybridPipeParallel(TestMultipleGpus):
    def test_hybrid_parallel_transformer_unbalanced_data(self):
        print(f"pwd {os.getcwd()}")
        self.run_mnist_2gpu('hybrid_parallel_pp_transformer_save.py')
        self.run_mnist_2gpu(
            'hybrid_parallel_pp_transformer_save_with_virtual_stage.py'
        )
        # test pp adaptor
        dir1 = "./pp_transformer"
        p_config1 = ParallelConfig(mp=1, pp=2, vpp=1, sharding=1)
        dir2 = "./pp_transformer_vp"
        p_config2 = ParallelConfig(mp=1, pp=2, vpp=2, sharding=1)

        pp_to_vp = PipeLineModelAdaptor(
            src_parallel_config=p_config1,
            dst_parallel_config=p_config2,
            transformer_layer_num=8,
            segment_method="layer",
        )
        vp_to_pp = PipeLineModelAdaptor(
            src_parallel_config=p_config2,
            dst_parallel_config=p_config1,
            transformer_layer_num=8,
            segment_method="layer",
        )

        def check_converted_model(converted_model_dir, expected_model_dir):
            # for compatibility, converted_model_dir may contain more key than
            # expected model, which does not hinder model recovering
            for i in range(p_config1.pp):
                sub_converted_model_dir = (
                    "{}/mp_00_sharding_00_pp_{:0>2d}".format(
                        converted_model_dir, i
                    )
                )
                sub_expected_model_dir = (
                    "{}/mp_00_sharding_00_pp_{:0>2d}".format(
                        expected_model_dir, i
                    )
                )
                print(
                    f"converted_model_dir: {sub_converted_model_dir}; expected_model_dir: {sub_expected_model_dir}"
                )

                def check_names(dict_1, dict_2):
                    for (k, v) in dict_2.items():
                        self.AssertTrue(k in dict_1)
                        self.AssertEqual(
                            getattr(v, "name", ""),
                            getattr(dict_1[k], "name", ""),
                        )

                # check param
                params_1 = paddle.load(
                    f"{sub_converted_model_dir}/model.pdparams"
                )
                params_2 = paddle.load(
                    f"{sub_expected_model_dir}/model.pdparams"
                )
                check_names(params_1, params_2)
                del params_1
                del params_2
                # check opt
                opt_1 = paddle.load(
                    f"{sub_converted_model_dir}/model_state.pdopt"
                )
                opt_2 = paddle.load(
                    f"{sub_expected_model_dir}/model_state.pdopt"
                )
                check_names(opt_1, opt_2)
                # check master weight
                # check master wieghts
                if "master_weights" in opt_2:
                    self.AssertTrue("master_weights" in opt_1)
                    check_names(
                        opt_2["master_weights"], opt_1["master_weights"]
                    )

        # check pp to vp
        tmp_dir1 = "./tmp_vp"
        if not os.path.exists(tmp_dir1):
            os.makedirs(tmp_dir1)
        pp_to_vp.apply(dir1, tmp_dir1)
        check_converted_model(tmp_dir1, dir2)

        # check vp to pp
        tmp_dir2 = "./tmp_pp"
        if not os.path.exists(tmp_dir2):
            os.makedirs(tmp_dir2)
        vp_to_pp.apply(dir2, tmp_dir2)
        check_converted_model(tmp_dir2, dir1)

        # rm dirs
        shutil.rmtree(dir1, ignore_errors=True)
        shutil.rmtree(dir2, ignore_errors=True)
        shutil.rmtree(tmp_dir1, ignore_errors=True)
        shutil.rmtree(tmp_dir2, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
