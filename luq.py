from contextlib import contextmanager

import paddle
from paddle.autograd.py_layer import PyLayer


class scaled_layer_1(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = x * 3
        return y

    @staticmethod
    def backward(ctx, dy):
        dx = paddle.sin(dy)
        return dx

@contextmanager
def enable_to_static_guard(flag: bool):
    program_translator = paddle.jit.api.ProgramTranslator()
    original_flag_value = program_translator.enable_to_static
    program_translator.enable(flag)
    try:
        yield
    finally:
        program_translator.enable(original_flag_value)

@paddle.jit.to_static(full_graph=True)
def test_func(x):
    y = scaled_layer_1.apply(x)
    return y
    # y.mean().backward()


inp = paddle.randn([1, 32])
inp.stop_gradient = False
    
with enable_to_static_guard(True):
    result = test_func(inp)
    loss = result.mean()
    loss.backward()

print("ok...")
    # print(paddle.static.default_main_program())