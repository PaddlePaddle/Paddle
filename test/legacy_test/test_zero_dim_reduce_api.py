#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

# Note:
# 0D Tensor indicates that the tensor's dimension is 0
# 0D Tensor's shape is always [], numel is 1
# which can be created by paddle.rand([])

import unittest

import numpy as np

import paddle

reduce_api_list = [
    paddle.sum,
    paddle.mean,
    paddle.nansum,
    paddle.nanmean,
    paddle.median,
    paddle.nanmedian,
    paddle.min,
    paddle.max,
    paddle.amin,
    paddle.amax,
    paddle.prod,
    paddle.logsumexp,
    paddle.all,
    paddle.any,
    paddle.count_nonzero,
]


# Use to test zero-dim of reduce API
class TestReduceAPI(unittest.TestCase):
    def assertShapeEqual(self, out, target_tuple):
        if not paddle.framework.in_pir_mode():
            out_shape = list(out.shape)
        else:
            out_shape = out.shape
        self.assertEqual(out_shape, target_tuple)

    def test_dygraph_reduce(self):
        paddle.disable_static()
        for api in reduce_api_list:
            # 1) x is 0D
            if api in [paddle.all, paddle.any]:
                x = paddle.randint(0, 2, []).astype('bool')
            else:
                x = paddle.rand([])
            x.stop_gradient = False
            out = api(x, axis=None)

            out.retain_grads()
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(out.shape, [])
            if api not in [paddle.count_nonzero]:
                np.testing.assert_allclose(out.numpy(), x.numpy())

            if api not in [paddle.median, paddle.nanmedian]:
                out_empty_list = api(x, axis=[])
                self.assertEqual(out_empty_list, out)
                self.assertEqual(out_empty_list.shape, [])

            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(out.grad.shape, [])
                np.testing.assert_allclose(x.grad.numpy(), np.array(1.0))
                np.testing.assert_allclose(out.grad.numpy(), np.array(1.0))

            out1 = api(x, axis=0)
            self.assertEqual(out1.shape, [])
            self.assertEqual(out1, out)
            out1.backward()

            out2 = api(x, axis=-1)
            self.assertEqual(out2.shape, [])
            self.assertEqual(out2, out)
            out2.backward()

            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                np.testing.assert_allclose(x.grad.numpy(), np.array(3.0))

            # 2) x is 1D, axis=0, reduce to 0D
            if api in [paddle.all, paddle.any]:
                x = paddle.randint(0, 2, [5]).astype('bool')
            else:
                x = paddle.rand([5])
            x.stop_gradient = False
            out = api(x, axis=0)
            out.retain_grads()
            out.backward()

            self.assertEqual(out.shape, [])
            if x.grad is not None:
                self.assertEqual(out.grad.shape, [])
                self.assertEqual(x.grad.shape, [5])

            # 3) x is ND, reduce to 0D
            if api in [paddle.all, paddle.any]:
                x = paddle.randint(0, 2, [3, 5]).astype('bool')
            else:
                x = paddle.rand([3, 5])
            x.stop_gradient = False
            out = api(x, axis=None)
            out.retain_grads()
            out.backward()

            self.assertEqual(out.shape, [])
            if x.grad is not None:
                self.assertEqual(out.grad.shape, [])
                self.assertEqual(x.grad.shape, [3, 5])

            # 4) x is ND, reduce to 0D, keepdim=True
            if api in [paddle.all, paddle.any]:
                x = paddle.randint(0, 2, [3, 5]).astype('bool')
            else:
                x = paddle.rand([3, 5])
            x.stop_gradient = False
            out = api(x, keepdim=True)
            out.retain_grads()
            out.backward()

            self.assertEqual(out.shape, [1, 1])
            if x.grad is not None:
                self.assertEqual(out.grad.shape, [1, 1])
                self.assertEqual(x.grad.shape, [3, 5])

        paddle.enable_static()

    def test_static_reduce_x_0D(self):
        paddle.enable_static()
        for api in reduce_api_list:
            main_prog = paddle.static.Program()
            exe = paddle.static.Executor()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                # 1) x is 0D
                if api in [paddle.all, paddle.any]:
                    x = paddle.randint(0, 2, []).astype('bool')
                else:
                    x = paddle.rand([])
                x.stop_gradient = False
                out = api(x, axis=None)
                grad_list = paddle.static.append_backward(
                    out, parameter_list=[x, out]
                )

                if api not in [paddle.median, paddle.nanmedian]:
                    out_empty_list = api(x, axis=[])
                    self.assertShapeEqual(out_empty_list, [])

                out1 = api(x, axis=0)
                self.assertShapeEqual(out1, [])

                out2 = api(x, axis=-1)
                self.assertShapeEqual(out2, [])

                fetch_list = [x, out]

                fetch_list.extend(
                    [
                        _grad
                        for _param, _grad in grad_list
                        if isinstance(
                            _grad,
                            (paddle.pir.Value, paddle.base.framework.Variable),
                        )
                    ]
                )
                res = exe.run(main_prog, fetch_list=fetch_list)

                for res_data in res:
                    self.assertEqual(res_data.shape, ())
                if api not in [paddle.count_nonzero]:
                    np.testing.assert_allclose(res[0], res[1])

                if len(res) > 3:
                    np.testing.assert_allclose(res[-2], np.array(1.0))
                    np.testing.assert_allclose(res[-1], np.array(1.0))
                if len(res) > 2:
                    np.testing.assert_allclose(res[-1], np.array(1.0))

    def test_static_reduce_ND_0D(self):
        paddle.enable_static()
        for api in reduce_api_list:
            main_prog = paddle.static.Program()
            exe = paddle.static.Executor()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                # 2) x is ND, reduce to 0D
                if api in [paddle.all, paddle.any]:
                    x = paddle.randint(0, 2, [3, 5]).astype('bool')
                else:
                    x = paddle.rand([3, 5])
                x.stop_gradient = False
                out = api(x, axis=None)
                grad_list = paddle.static.append_backward(
                    out, parameter_list=[out, x]
                )

                fetch_list = [out]
                fetch_list.extend(
                    [
                        _grad
                        for _param, _grad in grad_list
                        if isinstance(
                            _grad,
                            (paddle.pir.Value, paddle.base.framework.Variable),
                        )
                    ]
                )

                res = exe.run(main_prog, fetch_list=fetch_list)
                self.assertEqual(res[0].shape, ())
                if len(res) > 1:
                    self.assertEqual(res[1].shape, ())
                if len(res) > 2:
                    self.assertEqual(res[2].shape, (3, 5))

    def test_static_reduce_x_1D(self):
        paddle.enable_static()
        for api in reduce_api_list:
            main_prog = paddle.static.Program()
            exe = paddle.static.Executor()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                # 3) x is 1D, axis=0, reduce to 0D
                if api in [paddle.all, paddle.any]:
                    x = paddle.randint(0, 2, [5]).astype('bool')
                else:
                    x = paddle.rand([5])
                x.stop_gradient = False
                out = api(x, axis=0)
                grad_list = paddle.static.append_backward(
                    out, parameter_list=[out, x]
                )

                fetch_list = [out]
                fetch_list.extend(
                    [
                        _grad
                        for _param, _grad in grad_list
                        if isinstance(
                            _grad,
                            (paddle.pir.Value, paddle.base.framework.Variable),
                        )
                    ]
                )

                res = exe.run(main_prog, fetch_list=fetch_list)
                self.assertEqual(res[0].shape, ())
                if len(res) > 1:
                    self.assertEqual(res[1].shape, ())
                if len(res) > 2:
                    self.assertEqual(res[2].shape, (5,))

        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
