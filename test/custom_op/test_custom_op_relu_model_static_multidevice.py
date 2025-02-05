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
import subprocess
import tempfile
import unittest


class TestCustomOpReluModelStaticMultiDevice(unittest.TestCase):
    def install_custom_op(self):
        cmds = [
            "python",
            "setup_for_static_multidevice_test.py",
            "install",
        ]
        p = subprocess.run(cmds)
        assert p.returncode == 0, f"Install Custom Op: Failed: {p}"

    def setUp(self):
        self.fleet_log_dir = tempfile.TemporaryDirectory()
        self.model_dir = tempfile.TemporaryDirectory()
        self.output_log_dir = tempfile.TemporaryDirectory()
        self.install_custom_op()

    def train(self, use_custom_op: bool = True):
        cmds = [
            "python",
            "-m",
            "paddle.distributed.launch",
        ]
        cmds += ["--log_dir", self.fleet_log_dir.name]
        cmds += ["custom_op_multidevice_model_train.py"]
        cmds += ["--output_dir", self.output_log_dir.name]
        cmds += ["--model_dir", self.model_dir.name]
        if use_custom_op:
            cmds += ["--use_custom_op"]
        cmds += ["--train_mode"]
        p = subprocess.run(cmds)
        assert p.returncode == 0, f"Fleet train: Failed: {p}"

    def eval(self, use_custom_op: bool = True):
        cmds = [
            "python",
            "-m",
            "paddle.distributed.launch",
        ]
        cmds += ["--log_dir", self.fleet_log_dir.name]
        cmds += ["custom_op_multidevice_model_train.py"]
        cmds += ["--output_dir", self.output_log_dir.name]
        cmds += ["--model_dir", self.model_dir.name]
        if use_custom_op:
            cmds += ["--use_custom_op"]
        p = subprocess.run(cmds)
        assert p.returncode == 0, f"Fleet eval: Failed: {p}"

    def tearDown(self):
        self.fleet_log_dir.cleanup()
        self.model_dir.cleanup()
        self.output_log_dir.cleanup()

    def test_train_and_eval(self):
        import paddle

        if not paddle.framework.use_pir_api():
            self.train(use_custom_op=True)
            self.train(use_custom_op=False)

            import numpy as np

            count = 0
            if paddle.framework.core.is_compiled_with_cuda():
                count = paddle.framework.core.get_cuda_device_count()
            elif paddle.framework.core.is_compiled_with_xpu():
                count = paddle.framework.core.get_xpu_device_count()
            assert (
                count > 1
            ), "TestCustomOpReluModelStaticMultiDevice needs at least two devices"

            for id in range(count):
                loss_custom = np.load(
                    os.path.join(
                        self.output_log_dir.name, f'train_{id}_{True}.npz'
                    )
                )
                loss_origin = np.load(
                    os.path.join(
                        self.output_log_dir.name,
                        f'train_{id}_{False}.npz',
                    )
                )
                np.testing.assert_array_equal(
                    loss_custom['losses'], loss_origin['losses']
                )
                np.testing.assert_array_equal(
                    loss_custom['relu_out1_list'], loss_origin['relu_out1_list']
                )
                np.testing.assert_array_equal(
                    loss_custom['relu_out2_list'], loss_origin['relu_out2_list']
                )

            self.eval(use_custom_op=True)
            self.eval(use_custom_op=False)
            for id in range(count):
                loss_custom = np.load(
                    os.path.join(
                        self.output_log_dir.name, f'eval_{id}_{True}.npz'
                    )
                )
                loss_origin = np.load(
                    os.path.join(
                        self.output_log_dir.name, f'eval_{id}_{False}.npz'
                    )
                )
                np.testing.assert_array_equal(
                    loss_custom['losses'], loss_origin['losses']
                )
                np.testing.assert_array_equal(
                    loss_custom['relu_out1_list'], loss_origin['relu_out1_list']
                )
                np.testing.assert_array_equal(
                    loss_custom['relu_out2_list'], loss_origin['relu_out2_list']
                )


if __name__ == '__main__':
    unittest.main()
