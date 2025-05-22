import paddle
from paddle.incubate.nn.functional import moe_combine

x = paddle.arange(1, 16).view((5, 3)).astype("float32") # [[1,2,3], [4,5,6], ..., [13,14,15]]
x.stop_gradient = False

# 组合权重（手动构造）, 数据类型需要与x相同
combine_weights = paddle.to_tensor([
[0.7, 0.3],
[0.6, 0.4],
[0.5, 0.5],
[0.4, 0.6],
[0.2, 0.8]
], stop_gradient=False)

# 分散索引 仅支持int32
scatter_index = paddle.to_tensor([
[0, 1, 2, 3, 4],
[0, 1, 2, 3, 4]
], dtype="int32", stop_gradient=False)

y = moe_combine(x, combine_weights, scatter_index)
print("\n##########forward output##########\n")
print(y)
print(f"x.grad: {x.grad,}, combine_weights.grad: {combine_weights.grad}, scatter_index.grad: {scatter_index.grad}")
y.backward()
print("\n##########backward output##########\n")
print(f"x.grad: {x.grad}\n combine_weights.grad: {combine_weights.grad}\n scatter_index.grad: {scatter_index.grad}")