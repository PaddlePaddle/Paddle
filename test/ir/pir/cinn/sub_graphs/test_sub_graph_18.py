# repo: PaddleClas
# model: ppcls^configs^ImageNet^Inception^InceptionV4
# api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.tensor.manipulation.squeeze||api:paddle.nn.functional.common.dropout||api:paddle.nn.functional.common.linear
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[1536, 1000],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[1000],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [22, 1536, 8, 8], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.pooling.adaptive_avg_pool2d(var_0, output_size=1, data_format='NCHW', name=None)
        var_2 = paddle.tensor.manipulation.squeeze(var_1, axis=[2, 3])
        var_3 = paddle.nn.functional.common.dropout(var_2, p=0.2, axis=None, training=True, mode='downscale_in_infer', name=None)
        var_4 = paddle.nn.functional.common.linear(x=var_3, weight=self.parameter_0, bias=self.parameter_1, name=None)
        return var_4


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 1536, 8, 8], dtype=paddle.float32),
        )
        self.net = LayerCase()
    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs
    # NOTE prim + cinn lead to error
    # NOTE output mismatch with prim
    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(self.net, to_static=True, with_prim=False, with_cinn=False)
        for st, cinn in zip(paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()