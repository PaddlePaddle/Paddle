import paddle
import torch
import numpy as np

torch.set_printoptions(precision=8)

# shape = [2, 3, 10]
# np_x = np.random.random(size=shape).astype("float32") - 0.5
# # np_dout = np.random.random(size=shape).astype("float32") - 0.5

# p_x = paddle.to_tensor(np_x).astype("bfloat16").astype("float32")
# p_x.stop_gradient = False
# # p_dout = paddle.to_tensor(np_dout).astype("bfloat16").astype("float32")
# # p_dout.stop_gradient = False

# p_max = paddle.max(p_x, axis=-1)
# p_max = paddle.repeat_interleave(p_max, repeats=shape[-1], axis=-1).reshape(shape)

# p_sub = paddle.subtract(p_x, p_max)

# p_exp = paddle.exp(p_sub)

# p_sum = paddle.sum(p_exp, axis=-1)
# p_sum = paddle.repeat_interleave(p_sum, repeats=shape[-1], axis=-1).reshape(shape)
# p_div = paddle.divide(p_exp, p_sum).astype("bfloat16")
# print(p_div)

# out = paddle.nn.functional.softmax(p_x, axis=-1, dtype="bfloat16")
# print(out)
# # out_grads = paddle.grad(out, p_x)
# # print(out_grads)


def my_softmax(x):
    x = x.astype("float32")
    shape = x.shape
    p_max = paddle.max(x, axis=-1)
    p_max = paddle.repeat_interleave(p_max, repeats=shape[-1], axis=-1).reshape(shape)

    p_sub = paddle.subtract(x, p_max)
    p_exp = paddle.exp(p_sub)
    p_sum = paddle.sum(p_exp, axis=-1)
    p_sum = paddle.repeat_interleave(p_sum, repeats=shape[-1], axis=-1).reshape(shape)
    p_div = paddle.divide(p_exp, p_sum).astype("bfloat16")
    return p_div

shape = [2, 3, 2]
np_x = np.random.random(size=shape).astype("float32") - 0.5
p_x = paddle.to_tensor(np_x).astype("bfloat16")
p_x.stop_gradient = False
print(p_x)
p_a = paddle.to_tensor(p_x)
p_a.stop_gradient = False
p_b = paddle.to_tensor(p_x)
p_b.stop_gradient = False
# p_c = paddle.to_tensor(p_x)
# p_c.stop_gradient = False
t_x = torch.tensor(
            np_x,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
print(t_x)

out = my_softmax(p_a)
# print(out)
out_g = paddle.randn(shape, "bfloat16")
paddle.autograd.backward(out, out_g)
# print(p_a.grad)

p_out = paddle.nn.functional.softmax(p_b, axis=-1, dtype="bfloat16")
# print(p_out)
paddle.autograd.backward(p_out, out_g)
# print(p_b.grad)

# t_out = torch.nn.functional.softmax(
#     t_x, dim=-1, dtype=torch.bfloat16
# )
# print(t_out)

# np.testing.assert_allclose(
#     t_out.to(dtype=torch.float32).detach().cpu().numpy(),
#     paddle.cast(p_out, "float32").numpy(),
#     rtol=1e-6,
#     atol=1e-6,
# )

# np.testing.assert_allclose(
#     out.numpy(),
#     p_out.numpy(),
#     rtol=1e-6,
#     atol=1e-6,
# )
# np.testing.assert_allclose(
#     p_a.grad.astype("float32").numpy(),
#     p_b.grad.astype("float32").numpy(),
#     rtol=1e-4,
#     atol=1e-4,
# )