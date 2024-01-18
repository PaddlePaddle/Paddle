# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^LeViT^LeViT_128
# api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.split,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.transpose,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,api||paddle.tensor.manipulation.concat,api||paddle.tensor.linalg.transpose,method||reshape,api||paddle.tensor.linalg.matmul,method||__mul__,method||__add__,api||paddle.nn.functional.activation.softmax,api||paddle.tensor.linalg.matmul,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape
import unittest

import numpy as np

import paddle


class SIR108(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_1886 = self.create_parameter(
            shape=[12, 16],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_1877,  # (shape: [10, 16, 768], dtype: paddle.float32, stop_gradient: False)
        var_1887,  # (shape: [16, 16], dtype: paddle.int64, stop_gradient: True)
    ):
        var_1878 = paddle.tensor.manipulation.reshape(
            var_1877, [10, 16, 12, 64]
        )
        out = paddle.tensor.manipulation.split(var_1878, [16, 16, 32], axis=3)
        var_1879 = out[0]
        var_1880 = out[1]
        var_1881 = out[2]
        var_1882 = paddle.tensor.linalg.transpose(var_1879, perm=[0, 2, 1, 3])
        var_1883 = paddle.tensor.linalg.transpose(var_1880, perm=[0, 2, 1, 3])
        var_1884 = paddle.tensor.linalg.transpose(var_1881, perm=[0, 2, 1, 3])
        var_1885 = paddle.tensor.linalg.transpose(var_1883, perm=[0, 1, 3, 2])
        var_1888 = paddle.tensor.linalg.transpose(self.var_1886, (1, 0))
        var_1889 = var_1887.__getitem__(0)
        var_1890 = paddle.tensor.manipulation.gather(var_1888, var_1889)
        var_1891 = var_1887.__getitem__(1)
        var_1892 = paddle.tensor.manipulation.gather(var_1888, var_1891)
        var_1893 = var_1887.__getitem__(2)
        var_1894 = paddle.tensor.manipulation.gather(var_1888, var_1893)
        var_1895 = var_1887.__getitem__(3)
        var_1896 = paddle.tensor.manipulation.gather(var_1888, var_1895)
        var_1897 = var_1887.__getitem__(4)
        var_1898 = paddle.tensor.manipulation.gather(var_1888, var_1897)
        var_1899 = var_1887.__getitem__(5)
        var_1900 = paddle.tensor.manipulation.gather(var_1888, var_1899)
        var_1901 = var_1887.__getitem__(6)
        var_1902 = paddle.tensor.manipulation.gather(var_1888, var_1901)
        var_1903 = var_1887.__getitem__(7)
        var_1904 = paddle.tensor.manipulation.gather(var_1888, var_1903)
        var_1905 = var_1887.__getitem__(8)
        var_1906 = paddle.tensor.manipulation.gather(var_1888, var_1905)
        var_1907 = var_1887.__getitem__(9)
        var_1908 = paddle.tensor.manipulation.gather(var_1888, var_1907)
        var_1909 = var_1887.__getitem__(10)
        var_1910 = paddle.tensor.manipulation.gather(var_1888, var_1909)
        var_1911 = var_1887.__getitem__(11)
        var_1912 = paddle.tensor.manipulation.gather(var_1888, var_1911)
        var_1913 = var_1887.__getitem__(12)
        var_1914 = paddle.tensor.manipulation.gather(var_1888, var_1913)
        var_1915 = var_1887.__getitem__(13)
        var_1916 = paddle.tensor.manipulation.gather(var_1888, var_1915)
        var_1917 = var_1887.__getitem__(14)
        var_1918 = paddle.tensor.manipulation.gather(var_1888, var_1917)
        var_1919 = var_1887.__getitem__(15)
        var_1920 = paddle.tensor.manipulation.gather(var_1888, var_1919)
        var_1921 = paddle.tensor.manipulation.concat(
            [
                var_1890,
                var_1892,
                var_1894,
                var_1896,
                var_1898,
                var_1900,
                var_1902,
                var_1904,
                var_1906,
                var_1908,
                var_1910,
                var_1912,
                var_1914,
                var_1916,
                var_1918,
                var_1920,
            ]
        )
        var_1922 = paddle.tensor.linalg.transpose(var_1921, (1, 0))
        var_1923 = var_1922.reshape((0, 16, 16))
        var_1924 = paddle.tensor.linalg.matmul(var_1882, var_1885)
        var_1925 = var_1924.__mul__(0.25)
        var_1926 = var_1925.__add__(var_1923)
        var_1927 = paddle.nn.functional.activation.softmax(var_1926)
        var_1928 = paddle.tensor.linalg.matmul(var_1927, var_1884)
        var_1929 = paddle.tensor.linalg.transpose(var_1928, perm=[0, 2, 1, 3])
        var_1930 = paddle.tensor.manipulation.reshape(var_1929, [10, 16, 384])
        return var_1930


class TestSIR108(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[10, 16, 768], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[16, 16], dtype=paddle.int64),
        )
        self.net = SIR108()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
