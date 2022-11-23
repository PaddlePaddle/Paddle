# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os
import unittest
from test_dist_base import TestDistBase
import paddle

paddle.enable_static()


class TestDistMnistFleetSave(TestDistBase):

    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_fleet_api = True
        self._save_model = True

    def _rm_temp_files(self, dirname):
        fluid_model_path = os.path.join(dirname, 'fluid_persistables')
        fleet_model_path = os.path.join(dirname, 'fleet_persistables')
        fluid_infer_path = os.path.join(dirname, 'fluid_infer')
        fleet_infer_path = os.path.join(dirname, 'fleet_infer')
        fluid_model_path_2 = os.path.join(dirname, 'fluid_persistables_2')
        fleet_model_path_2 = os.path.join(dirname, 'fleet_persistables_2')
        fluid_infer_path_2 = os.path.join(dirname, 'fluid_infer_2')
        fleet_infer_path_2 = os.path.join(dirname, 'fleet_infer_2')

        shutil.rmtree(fluid_model_path)
        shutil.rmtree(fleet_model_path)
        shutil.rmtree(fluid_infer_path)
        shutil.rmtree(fleet_infer_path)
        shutil.rmtree(fluid_model_path_2)
        shutil.rmtree(fleet_model_path_2)
        shutil.rmtree(fluid_infer_path_2)
        shutil.rmtree(fleet_infer_path_2)

    def _test_saved_files(self, dirname):
        fluid_model_path = os.path.join(dirname, 'fluid_persistables')
        fluid_persistables = sorted(os.listdir(fluid_model_path))
        fleet_model_path = os.path.join(dirname, 'fleet_persistables')
        fleet_persistables = sorted(os.listdir(fleet_model_path))
        fluid_infer_path = os.path.join(dirname, 'fluid_infer')
        fluid_infer_files = sorted(os.listdir(fluid_infer_path))
        fleet_infer_path = os.path.join(dirname, 'fleet_infer')
        fleet_infer_files = sorted(os.listdir(fleet_infer_path))

        if len(fluid_persistables) != len(fleet_persistables):
            self._rm_temp_files(dirname)
            raise ValueError("Test Failed.")
        for i in range(len(fluid_persistables)):
            if fluid_persistables[i] != fleet_persistables[i]:
                self._rm_temp_files(dirname)
                raise ValueError("Test Failed.")

        if len(fluid_infer_files) != len(fleet_infer_files):
            self._rm_temp_files(dirname)
            raise ValueError("Test Failed.")
        for i in range(len(fluid_infer_files)):
            if fluid_infer_files[i] != fleet_infer_files[i]:
                self._rm_temp_files(dirname)
                raise ValueError("Test Failed.")
        self._rm_temp_files(dirname)
        return True

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={},
                         log_name=""):
        required_envs = self._get_required_envs(check_error_log, need_envs)

        tr0_losses, tr1_losses = self._run_cluster_nccl2(
            model_file,
            required_envs,
            update_method='nccl2',
            check_error_log=check_error_log,
            log_name=log_name)

        dirname = '/tmp'
        self._test_saved_files(dirname)

    def test_dist_train(self):
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place("dist_mnist.py", delta=1e-5)


if __name__ == "__main__":
    unittest.main()
