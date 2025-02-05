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

import numpy as np
import parameterized as param

import paddle
from paddle.base import core

RTOL = 1e-5
ATOL = 1e-8

PLACES = [('cpu', paddle.CPUPlace())] + (
    [('gpu', paddle.CUDAPlace(0))] if core.is_compiled_with_cuda() else []
)


def func_ref(func, *inputs):
    """ref func, just for convenience"""
    return func(*inputs)


test_list = [
    (paddle.atleast_1d, np.atleast_1d),
    (paddle.atleast_2d, np.atleast_2d),
    (paddle.atleast_3d, np.atleast_3d),
]


def generate_data(ndim, count=1, max_size=4, mix=False, dtype='int32'):
    """generate test data

    Args:
        ndim(int): dim of inputs
        count(int): input count for each dim
        max_size(int): max size for each dim
        mix(bool): mix data types or not, like a data list [123, np.array(123), paddle.to_tensor(123), ...]
        dtype(str): dtype

    Returns:
        a list of data like:
            [[data, dtype, shape, name], [data, dtype, shape, name] ... ]
    """
    rtn = []
    for d in range(ndim):
        data = [
            np.random.randint(
                0,
                255,
                size=[np.random.randint(1, max_size) for _ in range(d)],
                dtype=dtype,
            )
            for _ in range(count)
        ]

        if mix:

            def _mix_data(data, idx):
                if idx % 3 == 0:
                    return data.tolist()
                elif idx % 3 == 1:
                    return data
                elif idx % 3 == 2:
                    return paddle.to_tensor(data)

            # mix normal/numpy/tensor
            rtn.append(
                list(
                    zip(
                        *[
                            [
                                _mix_data(_data, idx),
                                str(_data.dtype),
                                _data.shape,
                                '{}d_{}_{}'.format(d, idx, 'mix'),
                            ]
                            for idx, _data in enumerate(data)
                        ]
                    )
                )
            )

        else:
            # normal
            rtn.append(
                list(
                    zip(
                        *[
                            [
                                _data.tolist(),
                                str(_data.dtype),
                                _data.shape,
                                '{}d_{}_{}'.format(d, idx, 'normal'),
                            ]
                            for idx, _data in enumerate(data)
                        ]
                    )
                )
            )
            # numpy
            rtn.append(
                list(
                    zip(
                        *[
                            [
                                _data,
                                str(_data.dtype),
                                _data.shape,
                                '{}d_{}_{}'.format(d, idx, 'numpy'),
                            ]
                            for idx, _data in enumerate(data)
                        ]
                    )
                )
            )
            # tensor
            rtn.append(
                list(
                    zip(
                        *[
                            [
                                paddle.to_tensor(_data),
                                str(_data.dtype),
                                _data.shape,
                                '{}d_{}_{}'.format(d, idx, 'tensor'),
                            ]
                            for idx, _data in enumerate(data)
                        ]
                    )
                )
            )
    return rtn


class BaseTest(unittest.TestCase):
    """Test in each `PLACES`, each `test_list`, and in `static/dygraph`"""

    def _test_static_api(
        self,
        inputs: list,
        dtypes: list,
        shapes: list,
        names: list,
    ):
        """Test `static`, convert `Tensor` to `numpy array` before feed into graph"""
        for device, place in PLACES:
            paddle.enable_static()
            paddle.set_device(device)

            for func, func_type in test_list:
                with paddle.static.program_guard(paddle.static.Program()):
                    x = []
                    feed = {}
                    for i in range(len(inputs)):
                        input = inputs[i]
                        shape = shapes[i]
                        dtype = dtypes[i]
                        name = names[i]

                        _x = paddle.static.data(name, shape, dtype)
                        _x.stop_gradient = False
                        x.append(_x)

                        # the data feeded should NOT be a Tensor
                        feed[name] = (
                            input.numpy()
                            if isinstance(input, paddle.Tensor)
                            else input
                        )

                    out = func(*x)

                    if len(inputs) == 1:
                        out.stop_gradient = False
                        y = x[0]
                        _out = out
                    else:
                        for o in out:
                            o.stop_gradient = False
                        y = x[0]
                        _out = out[0]

                    z = _out * 123

                    fetch_list = [out]
                    if paddle.framework.in_pir_mode():
                        grads = paddle.autograd.ir_backward.grad(z, y)
                        out_grad = grads[0]
                        fetch_list.append(out_grad)
                    else:
                        paddle.static.append_backward(z)
                        out_grad = y.grad_name
                        fetch_list.append(out_grad)

                    exe = paddle.static.Executor(place)
                    *res, res_grad = exe.run(feed=feed, fetch_list=fetch_list)

                    # not check old ir
                    if paddle.framework.in_pir_mode():
                        # convert grad value to bool if dtype is bool
                        grad_value = 123.0 if dtypes[0] != 'bool' else True
                        np.testing.assert_allclose(
                            res_grad, np.ones_like(y) * grad_value
                        )

                out_ref = func_ref(
                    func_type,
                    *[
                        (
                            input.numpy()
                            if isinstance(input, paddle.Tensor)
                            else input
                        )
                        for input in inputs
                    ],
                )

                if len(inputs) == 1:
                    out_ref = [out_ref]

                for n, p in zip(out_ref, res):
                    np.testing.assert_allclose(n, p, rtol=RTOL, atol=ATOL)

    def _test_dygraph_api(
        self,
        inputs: list,
        dtypes: list,
        shapes: list,
        names: list,
    ):
        """Test `dygraph`, and check grads"""
        for device, place in PLACES:
            paddle.disable_static(place)
            paddle.set_device(device)

            for func, func_type in test_list:
                out = func(*inputs)
                out_ref = func_ref(
                    func_type,
                    *[
                        (
                            input.numpy()
                            if isinstance(input, paddle.Tensor)
                            else input
                        )
                        for input in inputs
                    ],
                )

                for n, p in zip(out_ref, out):
                    np.testing.assert_allclose(
                        n, p.numpy(), rtol=RTOL, atol=ATOL
                    )

                # check grads
                if len(inputs) == 1:
                    out = [out]

                for y in out:
                    y.stop_gradient = False
                    z = y * 123
                    grads = paddle.grad(z, y)
                    self.assertTrue(len(grads), 1)
                    self.assertEqual(grads[0].dtype, y.dtype)
                    self.assertEqual(grads[0].shape, y.shape)


@param.parameterized_class(
    ('inputs', 'dtypes', 'shapes', 'names'),
    (generate_data(5, count=1, max_size=4, dtype='int32')),
)
class TestAtleastDim(BaseTest):
    """test dim from 0 to 5"""

    def test_all(self):
        self._test_dygraph_api(
            self.inputs, self.dtypes, self.shapes, self.names
        )
        self._test_static_api(self.inputs, self.dtypes, self.shapes, self.names)


@param.parameterized_class(
    ('inputs', 'dtypes', 'shapes', 'names'),
    (generate_data(5, count=3, max_size=4, dtype='int32')),
)
class TestAtleastDimMoreInputs(BaseTest):
    """test inputs of 3 tensors"""

    def test_all(self):
        self._test_dygraph_api(
            self.inputs, self.dtypes, self.shapes, self.names
        )
        self._test_static_api(self.inputs, self.dtypes, self.shapes, self.names)


@param.parameterized_class(
    ('inputs', 'dtypes', 'shapes', 'names'),
    (generate_data(5, count=5, max_size=4, mix=True, dtype='int32')),
)
class TestAtleastMixData(BaseTest):
    """test mix number/numpy/tensor"""

    def test_all(self):
        self._test_dygraph_api(
            self.inputs, self.dtypes, self.shapes, self.names
        )
        self._test_static_api(self.inputs, self.dtypes, self.shapes, self.names)


@param.parameterized_class(
    ('inputs', 'dtypes', 'shapes', 'names'),
    (
        (
            (
                123,
                np.array([123], dtype='int32'),
                paddle.to_tensor([[123]], dtype='int32'),
                [[[123]]],
                np.array([[[[123]]]], dtype='int32'),
                paddle.to_tensor([[[[[123]]]]], dtype='int32'),
            ),
            ('int32', 'int32', 'int32', 'int32', 'int32', 'int32'),
            ((), (1,), (1, 1), (1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1, 1)),
            (
                '0_mixdim',
                '1_mixdim',
                '2_mixdim',
                '3_mixdim',
                '4_mixdim',
                '5_mixdim',
            ),
        ),
    ),
)
class TestAtleastMixDim(BaseTest):
    """test mix dim"""

    def test_all(self):
        self._test_dygraph_api(
            self.inputs, self.dtypes, self.shapes, self.names
        )
        self._test_static_api(self.inputs, self.dtypes, self.shapes, self.names)


@param.parameterized_class(
    ('inputs', 'dtypes', 'shapes', 'names'),
    (
        (
            (
                paddle.to_tensor(True, dtype='bool'),
                paddle.to_tensor(0.1, dtype='float16'),
                paddle.to_tensor(0.1, dtype='float32'),
                paddle.to_tensor(0.1, dtype='float64'),
                paddle.to_tensor(1, dtype='int8'),
                paddle.to_tensor(1, dtype='int16'),
                paddle.to_tensor(1, dtype='int32'),
                paddle.to_tensor(1, dtype='int64'),
                paddle.to_tensor(1, dtype='uint8'),
                paddle.to_tensor(1 + 1j, dtype='complex64'),
                paddle.to_tensor(1 + 1j, dtype='complex128'),
                paddle.to_tensor(0.1, dtype='bfloat16'),
            ),
            (
                'bool',
                'float16',
                'float32',
                'float64',
                'int8',
                'int16',
                'int32',
                'int64',
                'uint8',
                'complex64',
                'complex128',
                'bfloat16',
            ),
            (
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                (),
            ),
            (
                '0_mixdtype',
                '1_mixdtype',
                '2_mixdtype',
                '3_mixdtype',
                '4_mixdtype',
                '5_mixdtype',
                '6_mixdtype',
                '7_mixdtype',
                '8_mixdtype',
                '9_mixdtype',
                '10_mixdtype',
                '11_mixdtype',
            ),
        ),
    ),
)
class TestAtleastMixDtypes(BaseTest):
    """test mix dtypes"""

    def test_all(self):
        self._test_dygraph_api(
            self.inputs, self.dtypes, self.shapes, self.names
        )
        self._test_static_api(self.inputs, self.dtypes, self.shapes, self.names)


@param.parameterized_class(
    ('inputs', 'dtypes', 'shapes', 'names'),
    (
        (((123, [123]),), ('int32',), ((),), ('0_combine',)),
        (
            ((np.array([123], dtype='int32'), [[123]]),),
            ('int32',),
            ((),),
            ('1_combine',),
        ),
        (
            (
                (
                    np.array([[123]], dtype='int32'),
                    paddle.to_tensor([[[123]]], dtype='int32'),
                ),
            ),
            ('int32',),
            ((),),
            ('2_combine',),
        ),
    ),
)
class TestAtleastErrorCombineInputs(BaseTest):
    """test combine inputs, like: `at_leastNd((x, y))`, where paddle treats like numpy"""

    def test_all(self):
        with self.assertRaises((ValueError, TypeError)):
            self._test_dygraph_api(
                self.inputs, self.dtypes, self.shapes, self.names
            )

        with self.assertRaises((ValueError, TypeError)):
            self._test_static_api(
                self.inputs, self.dtypes, self.shapes, self.names
            )


class TestAtleastAsTensorMethod(unittest.TestCase):
    def test_as_tensor_method(self):
        input = 123

        for device, place in PLACES:
            paddle.disable_static(place)
            paddle.set_device(device)

            tensor = paddle.to_tensor(input, place=place)

            out = tensor.atleast_1d()
            out_ref = np.atleast_1d(input)

            for n, p in zip(out_ref, out):
                np.testing.assert_allclose(n, p.numpy(), rtol=RTOL, atol=ATOL)

            out = tensor.atleast_2d()
            out_ref = np.atleast_2d(input)

            for n, p in zip(out_ref, out):
                np.testing.assert_allclose(n, p.numpy(), rtol=RTOL, atol=ATOL)

            out = tensor.atleast_3d()
            out_ref = np.atleast_3d(input)

            for n, p in zip(out_ref, out):
                np.testing.assert_allclose(n, p.numpy(), rtol=RTOL, atol=ATOL)


if __name__ == '__main__':
    unittest.main()
