#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from paddle import to_tensor
from paddle.nn.functional import triplet_margin_loss
from paddle.nn import ZeroPad2D
import paddle

def pairwise_distance_ref(x1, x2, p, eps, keepdim=False):
    # keepdim的用法
    x1_dim = len(x1.shape)
    x2_dim = len(x2.shape)
    output_dim = x1_dim if x1_dim > x2_dim else x2_dim
    innermost_dim = output_dim - 1
    return np.linalg.norm(x1 - x2 + eps, p, innermost_dim, keepdim)


def triplet_margin_loss_ref(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False, size_average=None,
                        reduce=None, reduction="mean"):
    ''' # type: (Tensor, Tensor, Tensor, float, float, float, bool, Optional[bool], Optional[bool], str) -> Tensor'''
    # dim 和 shape
    a_dim = len(anchor.shape)
    p_dim = len(positive.shape)
    n_dim = len(negative.shape)
    if not (a_dim == p_dim and p_dim == n_dim):
        print(f"All input should have same dim but got, {a_dim, p_dim, n_dim}  ")
    dist_p = pairwise_distance_ref(anchor, positive, p, eps)
    dist_n = pairwise_distance_ref(anchor, negative, p, eps)
    if swap:
        dist_swap = pairwise_distance_ref(positive, negative, p, eps)
        # min的用法
        dist_n = np.min(dist_n, dist_swap)
    loss = np.maximum(margin + dist_p - dist_n,0)
    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)
    elif reduction == 'batchmean':
        batch_size = np.shape(anchor)[0]
        loss = np.sum(loss) / batch_size
    return loss




class TestTripletMarginLossAPIError(unittest.TestCase):
    """
    test paddle.nn.functional.triplet_margin_loss error.
    """

    def setUp(self):
        """
        unsupport dtypes
        """
        self.shape = [4, 3, 224, 224]
        self.unsupport_dtypes = ['bool', 'int8', 'int32', 'int64']

    def test_unsupport_dtypes(self):
        """
        test unsupport dtypes.
        """
        for dtype in self.unsupport_dtypes:
            a = np.random.randint(-255, 255, size=self.shape)
            p = np.random.randint(-255, 255, size=self.shape)
            n = np.random.randint(-255, 255, size=self.shape)
            a_tensor = to_tensor(a).astype(dtype)
            p_tensor = to_tensor(p).astype(dtype)
            n_tensor = to_tensor(n).astype(dtype)
            self.assertRaises(TypeError, triplet_margin_loss, anchor=a_tensor, positive=p_tensor, negative=n_tensor)


class TestTripletMarginLossAPI(unittest.TestCase):
    """
    test paddle.nn.functional.triplet_margin_loss
    """

    def setUp(self):
        """
        support dtypes
        """
        self.shape = [4, 3, 224, 224]
        self.support_dtypes = ['float32', 'float64']

    def test_support_dtypes(self):
        """
        test support types
        """
        for dtype in self.support_dtypes:
            a = np.random.randint(-255, 255, size=self.shape).astype(dtype)
            p = np.random.randint(-255, 255, size=self.shape).astype(dtype)
            n = np.random.randint(-255, 255, size=self.shape).astype(dtype)


            expect_res = triplet_margin_loss_ref(a,p,n)
            a_tensor = to_tensor(a).astype(dtype)
            p_tensor = to_tensor(p).astype(dtype)
            n_tensor = to_tensor(n).astype(dtype)
            ret_res = triplet_margin_loss(a_tensor,p_tensor,n_tensor).numpy()
            self.assertTrue(np.allclose(expect_res, ret_res))

    # def test_support_pad2(self):
    #     """
    #     test the type of 'pad' is list.
    #     """
    #     pad = [1, 2, 3, 4]
    #     x = np.random.randint(-255, 255, size=self.shape)
    #     expect_res = np.pad(
    #         x, [[0, 0], [0, 0], [pad[2], pad[3]], [pad[0], pad[1]]])
    #
    #     x_tensor = to_tensor(x)
    #     ret_res = zeropad2d(x_tensor, pad).numpy()
    #     self.assertTrue(np.allclose(expect_res, ret_res))
    #
    # def test_support_pad3(self):
    #     """
    #     test the type of 'pad' is tuple.
    #     """
    #     pad = (1, 2, 3, 4)
    #     x = np.random.randint(-255, 255, size=self.shape)
    #     expect_res = np.pad(
    #         x, [[0, 0], [0, 0], [pad[2], pad[3]], [pad[0], pad[1]]])
    #
    #     x_tensor = to_tensor(x)
    #     ret_res = zeropad2d(x_tensor, pad).numpy()
    #     self.assertTrue(np.allclose(expect_res, ret_res))
    #
    # def test_support_pad4(self):
    #     """
    #     test the type of 'pad' is paddle.Tensor.
    #     """
    #     pad = [1, 2, 3, 4]
    #     x = np.random.randint(-255, 255, size=self.shape)
    #     expect_res = np.pad(
    #         x, [[0, 0], [0, 0], [pad[2], pad[3]], [pad[0], pad[1]]])
    #
    #     x_tensor = to_tensor(x)
    #     pad_tensor = to_tensor(pad, dtype='int32')
    #     ret_res = zeropad2d(x_tensor, pad_tensor).numpy()
    #     self.assertTrue(np.allclose(expect_res, ret_res))


class TestZeroPad2DLayer(unittest.TestCase):
    """
    test nn.ZeroPad2D
    """

    def setUp(self):
        self.shape = [4, 3, 224, 224]
        self.pad = [2, 2, 4, 1]
        self.padLayer = ZeroPad2D(padding=self.pad)
        self.x = np.random.randint(-255, 255, size=self.shape)
        self.expect_res = np.pad(self.x,
                                 [[0, 0], [0, 0], [self.pad[2], self.pad[3]],
                                  [self.pad[0], self.pad[1]]])

    def test_layer(self):
        self.assertTrue(
            np.allclose(
                zeropad2d(to_tensor(self.x), self.pad).numpy(),
                self.padLayer(to_tensor(self.x))))


if __name__ == '__main__':
    unittest.main()
