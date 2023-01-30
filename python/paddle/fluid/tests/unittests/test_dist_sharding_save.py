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

<<<<<<< HEAD
import os
import shutil
import unittest

from test_dist_base import TestDistBase

=======
from __future__ import print_function
import shutil
import os
import unittest
from test_dist_base import TestDistBase
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle

paddle.enable_static()


class TestDistMnistFleetSave(TestDistBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_fleet_api = True
        self._sharding_save = True
        self._enforce_place = "GPU"

    def _rm_temp_files(self, dirname):
        shutil.rmtree(dirname)

    def _test_saved_files(self, dirname):

        sharding_save_files = sorted(os.listdir(dirname))

        check_files = [
<<<<<<< HEAD
            'fc_0.b_0',
            'fc_0.b_0_velocity_0',
            'fc_0.w_0',
            'fc_0.w_0_velocity_0',
            'fc_1.b_0',
            'fc_1.b_0_velocity_0',
            'fc_1.w_0',
            'fc_1.w_0_velocity_0',
            'fc_2.b_0',
            'fc_2.b_0_velocity_0',
            'fc_2.w_0',
            'fc_2.w_0_velocity_0',
            'learning_rate_0',
=======
            'fc_0.b_0', 'fc_0.b_0_velocity_0', 'fc_0.w_0',
            'fc_0.w_0_velocity_0', 'fc_1.b_0', 'fc_1.b_0_velocity_0',
            'fc_1.w_0', 'fc_1.w_0_velocity_0', 'fc_2.b_0',
            'fc_2.b_0_velocity_0', 'fc_2.w_0', 'fc_2.w_0_velocity_0',
            'learning_rate_0'
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ]

        if sharding_save_files != check_files:
            self._rm_temp_files(dirname)
            raise ValueError("Test Failed.")
        self._rm_temp_files(dirname)

        return True

<<<<<<< HEAD
    def check_with_place(
        self,
        model_file,
        delta=1e-3,
        check_error_log=True,
        need_envs={},
        log_name="",
    ):
=======
    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=True,
                         need_envs={},
                         log_name=""):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        required_envs = self._get_required_envs(check_error_log, need_envs)

        tr0_losses, tr1_losses = self._run_cluster_nccl2(
            model_file,
            required_envs,
            update_method='nccl2',
            check_error_log=check_error_log,
<<<<<<< HEAD
            log_name=log_name,
        )
=======
            log_name=log_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        dirname = './ut_sharding_save_model'
        self._test_saved_files(dirname)

    def test_dist_train(self):
        import paddle.fluid as fluid
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place("dist_sharding_save.py", delta=1e-5)


if __name__ == "__main__":
    unittest.main()
