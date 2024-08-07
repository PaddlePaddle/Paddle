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
import unittest

import paddle


class TestAdamWFP16XPU(unittest.TestCase):
    def test_tensor_scale_value(self):
        x = paddle.to_tensor([9.876, 5.432, 2.10987])
        # read default scale_value
        self.assertEqual(x.get_tensor().get_xpu_scale_value(), -1)
        # set scale_value
        x.get_tensor().set_xpu_scale_value(-1.25)
        # read modified scale_value
        self.assertEqual(x.get_tensor().get_xpu_scale_value(), -1.25)

    def _test_state_dict(self):
        xpu_adamw_moment_dtype = os.getenv(
            "xpu_adamw_moment_dtype", default="fp32"
        )
        if xpu_adamw_moment_dtype == "fp16":
            use_fp16 = True
        else:
            use_fp16 = False
        linear = paddle.nn.Linear(10, 10)
        inp = paddle.rand([10, 10], dtype="float32")
        out = linear(inp)
        loss = paddle.mean(out)

        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")

        adam = paddle.optimizer.AdamW(
            learning_rate=0.1,
            parameters=linear.parameters(),
            beta1=beta1,
            beta2=beta2,
            weight_decay=0.01,
        )
        out.backward()
        adam.step()

        # read scale_value in state dict
        state_dict_1 = adam.state_dict()
        if use_fp16:
            self.assertTrue(
                "linear_0.w_0_moment1_0.SCALE_VALUE" in state_dict_1
            )
            self.assertTrue(
                "linear_0.b_0_moment1_0.SCALE_VALUE" in state_dict_1
            )
        else:
            self.assertTrue(
                "linear_0.w_0_moment1_0.SCALE_VALUE" not in state_dict_1
            )
            self.assertTrue(
                "linear_0.b_0_moment1_0.SCALE_VALUE" not in state_dict_1
            )

        if not use_fp16:
            # do not need "overwrite" and "check overwritten value" below
            return

        # overwrite scale_value
        state_dict_1["linear_0.w_0_moment1_0.SCALE_VALUE"] = 0.75
        state_dict_1["linear_0.b_0_moment1_0.SCALE_VALUE"] = 12.3125
        adam.set_state_dict(state_dict_1)

        # check overwritten value
        state_dict_2 = adam.state_dict()
        self.assertTrue("linear_0.w_0_moment1_0.SCALE_VALUE" in state_dict_2)
        self.assertTrue("linear_0.b_0_moment1_0.SCALE_VALUE" in state_dict_2)
        self.assertEqual(
            state_dict_2["linear_0.w_0_moment1_0.SCALE_VALUE"], 0.75
        )
        self.assertEqual(
            state_dict_2["linear_0.b_0_moment1_0.SCALE_VALUE"], 12.3125
        )

    def test_state_dict(self):
        os.environ["xpu_adamw_moment_dtype"] = "fp16"
        self._test_state_dict()
        os.environ["xpu_adamw_moment_dtype"] = "fp32"
        self._test_state_dict()


if __name__ == '__main__':
    paddle.disable_static()
    unittest.main()
