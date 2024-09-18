#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle


class A:
    pass


def function(x: A) -> A:
    t: A = A()
    return 2 * x


def fn_annotation_assign_with_value(x: paddle.Tensor):
    if x:
        y: list[paddle.Tensor] = [x + 1]
    else:
        y: list[paddle.Tensor] = [x - 1]
    return y


def fn_annotation_assign_without_value(x: paddle.Tensor):
    if x:
        y: list[paddle.Tensor]
        y = [x + 1]
    else:
        y = [x - 1]
    return y


class TestTypeHints(Dy2StTestBase):
    def setUp(self):
        self.x = np.zeros(shape=(1), dtype=np.int32)
        self._init_dyfunc()

    def _init_dyfunc(self):
        self.dyfunc = function

    def _run_static(self):
        return self._run(to_static=True)

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run(self, to_static):
        # Set the input of dyfunc to Tensor
        tensor_x = paddle.to_tensor(self.x)
        if to_static:
            ret = paddle.jit.to_static(self.dyfunc)(tensor_x)
        else:
            ret = self.dyfunc(tensor_x)
        if hasattr(ret, "numpy"):
            return ret.numpy()
        else:
            return ret

    def test_ast_to_func(self):
        static_numpy = self._run_static()
        dygraph_numpy = self._run_dygraph()
        np.testing.assert_allclose(dygraph_numpy, static_numpy, rtol=1e-05)


class TestAnnAssign(Dy2StTestBase):
    def assert_fn_dygraph_and_static_unified(self, dygraph_fn, x):
        static_fn = paddle.jit.to_static(dygraph_fn)
        dygraph_fn = dygraph_fn
        static_res = static_fn(x)
        dygraph_res = dygraph_fn(x)
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)

    def test_ann_assign_with_value(self):
        self.assert_fn_dygraph_and_static_unified(
            fn_annotation_assign_with_value, paddle.to_tensor(1)
        )

    def test_ann_assign_without_value(self):
        self.assert_fn_dygraph_and_static_unified(
            fn_annotation_assign_without_value, paddle.to_tensor(1)
        )


if __name__ == '__main__':
    unittest.main()
