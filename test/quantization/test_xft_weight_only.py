import time
import paddle
from paddle.nn.quant import weight_only_linear
from paddle._C_ops import xft_weight_only_linear, xft_weight_quantize

BITS = "int8"
paddle.seed(82)
x = paddle.randn([1, 1, 4096], dtype=paddle.float32).cpu()
weight = paddle.randn([4096, 12288], dtype='float32').cpu() / 10
# for i in range(weight.shape[0]):
#     weight[i] = i

weight_quant, scale, zero_point = xft_weight_quantize(weight, "weight_only_" + BITS)
# import pdb;pdb.set_trace()

for i in range(5):
    out = xft_weight_only_linear(x, weight_quant, None, scale, zero_point, BITS)
    ref_out = paddle.matmul(x=x, y=weight)

TIMES = 100
s_time = time.perf_counter()
for i in range(TIMES):
    out = xft_weight_only_linear(x, weight_quant, None, scale, zero_point, BITS)
e_time = time.perf_counter()
ellapse = (e_time - s_time) / TIMES * 1000
print(f"xft_weight_only_linear time: {ellapse:4.5f}")


s_time = time.perf_counter()
for i in range(TIMES):
    ref_out = paddle.matmul(x=x, y=weight)
e_time = time.perf_counter()
ellapse = (e_time - s_time) / TIMES * 1000
print(f"fp32 linear time:  {ellapse:4.5f}")
# import pdb;pdb.set_trace()
# weight_dequant = weight.T.astype(paddle.float32) * scale
ref_out = paddle.matmul(x=x, y=weight)

# print(out)
# print(ref_out)
print(out - ref_out)