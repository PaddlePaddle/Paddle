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

import functools
import unittest

import numpy as np

import paddle
from paddle.base import core

RTOL = 1e-5
ATOL = 1e-8

DTYPE_ALL = [
    'float16',
    'float32',
    'float64',
    'int16',
    'int32',
    'int64',
    'int8',
    'uint8',
    'complex64',
    'complex128',
    'bfloat16',
    'bool',
]

DTYPE_DYGRAPH_SPLIT = [
    'float64',
    'float16',
    'float32',
    'bool',
    'uint8',
    'bfloat16',
    'int32',
    'int8',
    'int64',
]
DTYPE_STATIC_SPLIT = DTYPE_DYGRAPH_SPLIT

DTYPE_DYGRAPH_H_SPLIT = DTYPE_DYGRAPH_SPLIT
DTYPE_STATIC_H_SPLIT = DTYPE_STATIC_SPLIT

DTYPE_DYGRAPH_V_SPLIT = DTYPE_DYGRAPH_SPLIT
DTYPE_STATIC_V_SPLIT = DTYPE_STATIC_SPLIT

DTYPE_DYGRAPH_D_SPLIT = DTYPE_DYGRAPH_SPLIT
DTYPE_STATIC_D_SPLIT = DTYPE_STATIC_SPLIT

DTYPE_DYGRAPH_TENSOR_SPLIT = DTYPE_ALL
DTYPE_STATIC_TENSOR_SPLIT = DTYPE_ALL

PLACES = [paddle.CPUPlace()] + (
    [paddle.CUDAPlace(0)] if core.is_compiled_with_cuda() else []
)


def convert_num_or_sections(num_or_sections):
    # hsplit, vsplit, dsplit
    # Convert the num_or_sections in paddle to indices_or_sections in numpy
    # Do not support -1
    if isinstance(num_or_sections, int):
        indices_or_sections = num_or_sections
    else:
        indices_or_sections = np.cumsum(num_or_sections)[:-1]

    return indices_or_sections


def generate_data(shape, dtype='int64'):
    """generate test data

    Args:
        shape(list of int): shape of inputs
        dtype(str): dtype

    Returns:
        x, dtype, shape, name
    """
    return {
        # bfloat16 convert to uint16 for numpy
        'x': np.random.randint(0, 255, size=shape).astype(
            dtype if dtype != 'bfloat16' else 'uint16'
        ),
        'dtype': dtype,
        'shape': shape,
        'name': f'{shape}_{dtype}',
    }


class BaseTest(unittest.TestCase):
    """Test in each `PLACES` and in `static/dygraph`"""

    def _test_static_api(
        self,
        func_paddle,
        func_numpy,
        x,
        dtype,
        shape,
        name,
        split_paddle,
        split_numpy,
    ):
        """Test `static`

        Args:
            func_paddle: `hsplit`, `vsplit`, `dsplit`, `tensor_split`
            func_numpy: `hsplit`, `vsplit`, `dsplit`, `array_split`
            x: input tensor
            dtype: input tensor's dtype
            shape: input tensor's shape
            name: input tensor's name
            split_paddle: num_or_sections or indices_or_sections in paddle
            split_numpy: `hsplit`, `vsplit`, `dsplit` should convert num_or_sections in paddle to indices_or_sections in numpy.
                For test error, `split_numpy` is None and skip compare result, ensure the error only raised from paddle.
        """
        paddle.enable_static()

        for place in PLACES:
            program = paddle.static.Program()
            exe = paddle.static.Executor(place)

            with paddle.static.program_guard(program):
                input = paddle.static.data(name, shape, dtype)
                feed = {name: x}

                out = func_paddle(input, split_paddle)
                res = exe.run(feed=feed, fetch_list=[out])

                if split_numpy is not None:
                    out_ref = func_numpy(x, split_numpy)

                    for n, p in zip(out_ref, res):
                        np.testing.assert_allclose(n, p, rtol=RTOL, atol=ATOL)

    def _test_dygraph_api(
        self,
        func_paddle,
        func_numpy,
        x,
        dtype,
        shape,
        name,
        split_paddle,
        split_numpy,
    ):
        """Test `dygraph`, and check grads"""
        paddle.disable_static()

        for place in PLACES:
            out = func_paddle(paddle.to_tensor(x).astype(dtype), split_paddle)

            if split_numpy is not None:
                out_ref = func_numpy(x, split_numpy)

                for n, p in zip(out_ref, out):
                    np.testing.assert_allclose(
                        n, p.numpy(), rtol=RTOL, atol=ATOL
                    )

                # check grads for the first tensor
                out = out[0]

                for y in out:
                    y.stop_gradient = False
                    z = y * 123
                    grads = paddle.grad(z, y)
                    self.assertTrue(len(grads), 1)
                    self.assertEqual(grads[0].dtype, y.dtype)
                    self.assertEqual(grads[0].shape, y.shape)

    def _test_all(
        self,
        kwargs,
        dtype_not_supported_in_dygraph=False,
        dtype_not_supported_in_static=False,
        dtype='',
    ):
        if dtype_not_supported_in_dygraph:
            # CAUTION: raise RuntimeError instead of TypeError!
            with self.assertRaises(RuntimeError):
                self._test_dygraph_api(
                    self.func_paddle, self.func_numpy, **kwargs
                )
        else:
            self._test_dygraph_api(self.func_paddle, self.func_numpy, **kwargs)

        if dtype_not_supported_in_static:
            with self.assertRaises(TypeError):
                self._test_static_api(
                    self.func_paddle, self.func_numpy, **kwargs
                )
        else:
            self._test_static_api(self.func_paddle, self.func_numpy, **kwargs)


class TestHSplit(BaseTest):
    def setUp(self):
        self.func_paddle = paddle.hsplit
        self.func_numpy = np.hsplit

    def test_split_dim(self):
        x = generate_data([6])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})

        self._test_all(
            {
                **x,
                'split_paddle': [2, 4],
                'split_numpy': convert_num_or_sections([2, 4]),
            }
        )
        self._test_all(
            {
                **x,
                'split_paddle': (2, 1, 3),
                'split_numpy': convert_num_or_sections((2, 1, 3)),
            }
        )
        self._test_all({**x, 'split_paddle': [1, -1, 3], 'split_numpy': [1, 3]})

        x = generate_data([4, 6])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})

        self._test_all(
            {
                **x,
                'split_paddle': [2, 4],
                'split_numpy': convert_num_or_sections([2, 4]),
            }
        )
        self._test_all(
            {
                **x,
                'split_paddle': (2, 1, 3),
                'split_numpy': convert_num_or_sections((2, 1, 3)),
            }
        )
        self._test_all({**x, 'split_paddle': [1, -1, 3], 'split_numpy': [1, 3]})

        x = generate_data([4, 6, 3])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})

        self._test_all(
            {
                **x,
                'split_paddle': [2, 4],
                'split_numpy': convert_num_or_sections([2, 4]),
            }
        )
        self._test_all(
            {
                **x,
                'split_paddle': (2, 1, 3),
                'split_numpy': convert_num_or_sections((2, 1, 3)),
            }
        )
        self._test_all({**x, 'split_paddle': [1, -1, 3], 'split_numpy': [1, 3]})

    def test_dtype(self):
        for dtype in DTYPE_ALL:
            self._test_all(
                {
                    **generate_data([6], dtype=dtype),
                    'split_paddle': 3,
                    'split_numpy': 3,
                },
                dtype not in DTYPE_DYGRAPH_H_SPLIT,
                dtype not in DTYPE_STATIC_H_SPLIT,
                dtype,
            )

    def test_error_dim(self):
        # test 0-d
        x = generate_data([])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

    def test_error_split(self):
        x = generate_data([5])

        # test not evenly split
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 2, 'split_numpy': None})

        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 0, 'split_numpy': None})

        # test bad split sections
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': [2, 1], 'split_numpy': None})

        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': [2, 4], 'split_numpy': None})

        # test more than one `-1`
        with self.assertRaises(ValueError):
            self._test_all(
                {**x, 'split_paddle': [-1, 2, -1], 'split_numpy': None}
            )


class TestVSplit(BaseTest):
    def setUp(self):
        self.func_paddle = paddle.vsplit
        self.func_numpy = np.vsplit

    def test_split_dim(self):
        x = generate_data([6, 4])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})

        self._test_all(
            {
                **x,
                'split_paddle': [2, 4],
                'split_numpy': convert_num_or_sections([2, 4]),
            }
        )
        self._test_all(
            {
                **x,
                'split_paddle': (2, 1, 3),
                'split_numpy': convert_num_or_sections((2, 1, 3)),
            }
        )
        self._test_all({**x, 'split_paddle': [1, -1, 3], 'split_numpy': [1, 3]})

        x = generate_data([6, 4, 3])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})

        self._test_all(
            {
                **x,
                'split_paddle': [2, 4],
                'split_numpy': convert_num_or_sections([2, 4]),
            }
        )
        self._test_all(
            {
                **x,
                'split_paddle': (2, 1, 3),
                'split_numpy': convert_num_or_sections((2, 1, 3)),
            }
        )
        self._test_all({**x, 'split_paddle': [1, -1, 3], 'split_numpy': [1, 3]})

    def test_dtype(self):
        for dtype in DTYPE_ALL:
            self._test_all(
                {
                    **generate_data([6, 4], dtype=dtype),
                    'split_paddle': 3,
                    'split_numpy': 3,
                },
                dtype not in DTYPE_DYGRAPH_V_SPLIT,
                dtype not in DTYPE_STATIC_V_SPLIT,
                dtype,
            )

    def test_error_dim(self):
        # test 0-d
        x = generate_data([])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

        # test 1-d
        x = generate_data([6])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

    def test_error_split(self):
        x = generate_data([5, 4])

        # test not evenly split
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 2, 'split_numpy': None})

        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 0, 'split_numpy': None})

        # test bad split sections
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': [2, 1], 'split_numpy': None})

        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': [2, 4], 'split_numpy': None})

        # test more than one `-1`
        with self.assertRaises(ValueError):
            self._test_all(
                {**x, 'split_paddle': [-1, 2, -1], 'split_numpy': None}
            )


class TestDSplit(BaseTest):
    def setUp(self):
        self.func_paddle = paddle.dsplit
        self.func_numpy = np.dsplit

    def test_split_dim(self):
        x = generate_data([4, 3, 6])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})

        self._test_all(
            {
                **x,
                'split_paddle': [2, 4],
                'split_numpy': convert_num_or_sections([2, 4]),
            }
        )
        self._test_all(
            {
                **x,
                'split_paddle': (2, 1, 3),
                'split_numpy': convert_num_or_sections((2, 1, 3)),
            }
        )
        self._test_all({**x, 'split_paddle': [1, -1, 3], 'split_numpy': [1, 3]})

    def test_dtype(self):
        for dtype in DTYPE_ALL:
            self._test_all(
                {
                    **generate_data([4, 3, 6], dtype=dtype),
                    'split_paddle': 3,
                    'split_numpy': 3,
                },
                dtype not in DTYPE_DYGRAPH_D_SPLIT,
                dtype not in DTYPE_STATIC_D_SPLIT,
                dtype,
            )

    def test_error_dim(self):
        # test 0-d
        x = generate_data([])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

        # test 1-d
        x = generate_data([6])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

        # test 2-d
        x = generate_data([4, 6])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

    def test_error_split(self):
        x = generate_data([3, 6, 5])

        # test not evenly split
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 2, 'split_numpy': None})

        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 0, 'split_numpy': None})

        # test bad split sections
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': [2, 1], 'split_numpy': None})

        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': [2, 4], 'split_numpy': None})

        # test more than one `-1`
        with self.assertRaises(ValueError):
            self._test_all(
                {**x, 'split_paddle': [-1, 2, -1], 'split_numpy': None}
            )


class TestTensorSplit(BaseTest):
    def setUp(self):
        self.func_paddle = paddle.tensor_split
        self.func_numpy = np.array_split

    def test_split_dim(self):
        x = generate_data([6])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})
        self._test_all({**x, 'split_paddle': [2, 4], 'split_numpy': [2, 4]})
        self._test_all({**x, 'split_paddle': [2, 3], 'split_numpy': [2, 3]})
        self._test_all({**x, 'split_paddle': (2, 13), 'split_numpy': (2, 13)})
        self._test_all(
            {**x, 'split_paddle': [2, 4, 13], 'split_numpy': [2, 4, 13]}
        )

        # not evenly split
        x = generate_data([7])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})
        self._test_all({**x, 'split_paddle': [2, 4], 'split_numpy': [2, 4]})
        self._test_all({**x, 'split_paddle': [2, 3], 'split_numpy': [2, 3]})
        self._test_all({**x, 'split_paddle': (2, 13), 'split_numpy': (2, 13)})
        self._test_all(
            {**x, 'split_paddle': [2, 4, 13], 'split_numpy': [2, 4, 13]}
        )

        x = generate_data([4, 6])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})
        self._test_all({**x, 'split_paddle': [2, 4], 'split_numpy': [2, 4]})
        self._test_all({**x, 'split_paddle': [2, 3], 'split_numpy': [2, 3]})
        self._test_all({**x, 'split_paddle': (2, 13), 'split_numpy': (2, 13)})
        self._test_all(
            {**x, 'split_paddle': [2, 4, 13], 'split_numpy': [2, 4, 13]}
        )

        x = generate_data([4, 6, 3])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})
        self._test_all({**x, 'split_paddle': [2, 4], 'split_numpy': [2, 4]})
        self._test_all({**x, 'split_paddle': [2, 3], 'split_numpy': [2, 3]})
        self._test_all({**x, 'split_paddle': (2, 13), 'split_numpy': (2, 13)})
        self._test_all(
            {**x, 'split_paddle': [2, 4, 13], 'split_numpy': [2, 4, 13]}
        )

    def test_split_axis(self):
        # 1-d
        self.func_paddle = functools.partial(paddle.tensor_split, axis=0)
        self.func_numpy = functools.partial(np.array_split, axis=0)

        x = generate_data([7])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})
        self._test_all({**x, 'split_paddle': [2, 3], 'split_numpy': [2, 3]})
        self._test_all({**x, 'split_paddle': (2, 13), 'split_numpy': (2, 13)})
        self._test_all(
            {**x, 'split_paddle': [2, 4, 13], 'split_numpy': [2, 4, 13]}
        )

        # 2-d
        self.func_paddle = functools.partial(paddle.tensor_split, axis=1)
        self.func_numpy = functools.partial(np.array_split, axis=1)

        x = generate_data([4, 7])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})
        self._test_all({**x, 'split_paddle': [2, 3], 'split_numpy': [2, 3]})
        self._test_all({**x, 'split_paddle': (2, 13), 'split_numpy': (2, 13)})
        self._test_all(
            {**x, 'split_paddle': [2, 4, 13], 'split_numpy': [2, 4, 13]}
        )

        # 3-d
        self.func_paddle = functools.partial(paddle.tensor_split, axis=2)
        self.func_numpy = functools.partial(np.array_split, axis=2)

        x = generate_data([4, 4, 7])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})
        self._test_all({**x, 'split_paddle': [2, 3], 'split_numpy': [2, 3]})
        self._test_all({**x, 'split_paddle': (2, 13), 'split_numpy': (2, 13)})
        self._test_all(
            {**x, 'split_paddle': [2, 4, 13], 'split_numpy': [2, 4, 13]}
        )

        # n-d
        self.func_paddle = functools.partial(paddle.tensor_split, axis=3)
        self.func_numpy = functools.partial(np.array_split, axis=3)

        x = generate_data([4, 4, 4, 7])
        self._test_all({**x, 'split_paddle': 3, 'split_numpy': 3})
        self._test_all({**x, 'split_paddle': 2, 'split_numpy': 2})
        self._test_all({**x, 'split_paddle': [2, 3], 'split_numpy': [2, 3]})
        self._test_all({**x, 'split_paddle': (2, 13), 'split_numpy': (2, 13)})
        self._test_all(
            {**x, 'split_paddle': [2, 4, 13], 'split_numpy': [2, 4, 13]}
        )

    def test_dtype(self):
        self.func_paddle = functools.partial(paddle.tensor_split, axis=0)
        self.func_numpy = functools.partial(np.array_split, axis=0)

        for dtype in DTYPE_ALL:
            self._test_all(
                {
                    **generate_data([6], dtype=dtype),
                    'split_paddle': 3,
                    'split_numpy': 3,
                },
                dtype not in DTYPE_DYGRAPH_TENSOR_SPLIT,
                dtype not in DTYPE_STATIC_TENSOR_SPLIT,
                dtype,
            )

        self.func_paddle = functools.partial(paddle.tensor_split, axis=1)
        self.func_numpy = functools.partial(np.array_split, axis=1)

        for dtype in DTYPE_ALL:
            self._test_all(
                {
                    **generate_data([4, 6], dtype=dtype),
                    'split_paddle': 3,
                    'split_numpy': 3,
                },
                dtype not in DTYPE_DYGRAPH_TENSOR_SPLIT,
                dtype not in DTYPE_STATIC_TENSOR_SPLIT,
                dtype,
            )

        self.func_paddle = functools.partial(paddle.tensor_split, axis=2)
        self.func_numpy = functools.partial(np.array_split, axis=2)

        for dtype in DTYPE_ALL:
            self._test_all(
                {
                    **generate_data([4, 4, 6], dtype=dtype),
                    'split_paddle': 3,
                    'split_numpy': 3,
                },
                dtype not in DTYPE_DYGRAPH_TENSOR_SPLIT,
                dtype not in DTYPE_STATIC_TENSOR_SPLIT,
                dtype,
            )

    def test_error_dim(self):
        # axis 0
        self.func_paddle = functools.partial(paddle.tensor_split, axis=0)
        self.func_numpy = functools.partial(np.array_split, axis=0)

        # test 0-d
        x = generate_data([])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

        # axis 1
        self.func_paddle = functools.partial(paddle.tensor_split, axis=1)
        self.func_numpy = functools.partial(np.array_split, axis=1)

        # test 0-d
        x = generate_data([])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

        # test 1-d
        x = generate_data([6])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

        # axis 2
        self.func_paddle = functools.partial(paddle.tensor_split, axis=2)
        self.func_numpy = functools.partial(np.array_split, axis=2)

        # test 0-d
        x = generate_data([])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

        # test 1-d
        x = generate_data([6])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

        # test 2-d
        x = generate_data([4, 6])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 3, 'split_numpy': None})

    def test_error_split(self):
        x = generate_data([6])
        with self.assertRaises(ValueError):
            self._test_all({**x, 'split_paddle': 0, 'split_numpy': None})


if __name__ == '__main__':
    unittest.main()
