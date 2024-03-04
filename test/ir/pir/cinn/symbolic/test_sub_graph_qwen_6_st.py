# repo: llm_sub_grpah
# model: qwen
# method:__add__||method:astype||method:pow||method:mean||method:__add__||api:paddle.tensor.ops.rsqrt||method:__mul__||method:astype||method:__mul__
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[4096],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 64, 4096], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 64, 4096], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = var_0 + var_1
        var_3 = var_2.astype('float32')
        var_4 = var_3.pow(2)
        var_5 = var_4.mean(-1, keepdim=True)
        var_6 = var_5 + 1e-06
        var_7 = paddle.tensor.ops.rsqrt(var_6)
        var_8 = var_3 * var_7
        var_9 = var_8.astype('float32')
        var_10 = var_9 * self.parameter_0
        return var_10, var_2


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 64, 4096], dtype=paddle.float32),
        paddle.rand(shape=[1, 64, 4096], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 64, 4096]).astype('float32'),
        np.random.random(size=[1, 64, 4096]).astype('float32'),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_paddle_inputs()
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
    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(self.net, to_static=True, with_prim=True, with_cinn=False)
        for st, cinn in zip(paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()