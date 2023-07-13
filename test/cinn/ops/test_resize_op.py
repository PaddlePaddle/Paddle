#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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


# paddle resize is based on cv2 module
# This test requires cv2 module (pip3.6 install opencv_python==3.2.0.7)
# @OpTestTool.skip_if(not is_compiled_with_cuda(),
#                    "x86 test will be skipped due to timeout.")
# class TestResizeOp(OpTest):
#     def setUp(self):
#         self.init_case()

#     def init_case(self):
#         self.in_shape = [1,2,220,300]
#         self.inputs = {
#             "x":
#             (np.random.random(self.in_shape) * 255).astype('int32')
#         }
#         self.out_shape = [240, 240]
#         self.mode = "nearest"

#     def build_paddle_program(self, target):
#         #paddle resize only support [HWC] format.
#         input = self.inputs["x"].reshape(self.in_shape[1:4]).transpose([1,2,0]).astype('uint8')
#         out = F.resize(input, self.out_shape, self.mode)
#         out = paddle.to_tensor(out.transpose([2,0,1]).reshape(self.in_shape[0:2]+self.out_shape), dtype="int32", stop_gradient=False)
#         self.paddle_outputs = [out]

#     def build_cinn_program(self, target):
#         builder = NetBuilder("resize")
#         x = builder.create_input(
#             self.nptype2cinntype(self.inputs["x"].dtype),
#             self.inputs["x"].shape, "x")
#         out = builder.resize(x, self.out_shape, self.mode)
#         prog = builder.build()
#         res = self.get_cinn_output(
#             prog, target, [x], [self.inputs["x"]], [out], passes=[])
#         self.cinn_outputs = [res[0]]

#     def check_outputs_and_grads(self):
#         self.build_paddle_program(self.target)
#         self.build_cinn_program(self.target)
#         expect = self.paddle_outputs[0].numpy()
#         actual = self.cinn_outputs[0]

#         self.assertEqual(
#             expect.dtype,
#             actual.dtype,
#             msg=
#             "[{}] The output dtype different, which expect shape is {} but actual is {}."
#             .format(self._get_device(), expect.dtype, actual.dtype))
#         self.assertEqual(
#             expect.shape,
#             actual.shape,
#             msg=
#             "[{}] The output shape different, which expect shape is {} but actual is {}."
#             .format(self._get_device(), expect.shape, actual.shape))

#         is_allclose = np.allclose(
#                 expect,
#                 actual,
#                 atol=1)
#         error_message = "np.allclose(expect, actual, atol=1) checks error!"
#         self.assertTrue(is_allclose, msg=error_message)

#     def test_check_results(self):
#         self.check_outputs_and_grads()

# @OpTestTool.skip_if(not is_compiled_with_cuda(),
#                   "x86 test will be skipped due to timeout.")
# class TestResizeOp1(TestResizeOp):
#     def init_case(self):
#         self.in_shape = [1,2,220,300]
#         self.inputs = {
#             "x":
#             (np.random.random(self.in_shape) * 255).astype('int32')
#         }
#         self.out_shape = [4, 4]
#         self.mode = "bilinear"

# @OpTestTool.skip_if(not is_compiled_with_cuda(),
#                    "x86 test will be skipped due to timeout.")
# class TestResizeOp2(TestResizeOp):
#     def init_case(self):
#         self.in_shape = [1,2,220,300]
#         self.inputs = {
#             "x":
#             (np.random.random(self.in_shape) * 255).astype('int32')
#         }
#         self.out_shape = [4, 4]
#         self.mode = "bicubic"

# if __name__ == "__main__":
#     unittest.main()
