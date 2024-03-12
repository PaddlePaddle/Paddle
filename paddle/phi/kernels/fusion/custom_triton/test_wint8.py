import paddle
from paddle.nn.quant import weight_quantize
from paddle.nn.quant import weight_only_linear

from triton_ops import triton_wint8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

M = 128
N = 4096
K = 4096

activation = paddle.randn((M, K), dtype=paddle.float16)
original_weight = paddle.randn((K, N), dtype=paddle.float16)

perm_qweight, scale = weight_quantize(original_weight, algo="weight_only_int8")
bias = paddle.rand((N,), dtype=paddle.float16) * 10

no_perm_qweight = original_weight / scale.reshape([1, N])
no_perm_qweight = paddle.round(no_perm_qweight)
no_perm_qweight = paddle.clip(no_perm_qweight, min=-127, max=127)
no_perm_qweight = no_perm_qweight.astype("int8")

paddle.device.cuda.synchronize()

# 下面是paddle的cutlass代码
import datetime
for i in range(100):
    paddle_output = weight_only_linear(activation, perm_qweight, bias, scale)

paddle.device.cuda.synchronize()
starttime = datetime.datetime.now()

for i in range(100):
    paddle_output = weight_only_linear(activation, perm_qweight, bias, scale)

paddle.device.cuda.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("paddle cutlass The whoel end to end time : ", time_ms, "ms")



# 下面是triton的计算代码
no_perm_qweight = no_perm_qweight.transpose([1,0]).contiguous()

assert activation.is_contiguous()
assert no_perm_qweight.is_contiguous()
assert scale.is_contiguous()
no_perum_uint_qweight = no_perm_qweight.astype("int32")
no_perum_uint_qweight = no_perum_uint_qweight + 128
no_perum_uint_qweight = no_perum_uint_qweight.astype("uint8")

for i in range(100):
    triton_output = triton_wint8(
        activation,
        no_perum_uint_qweight,
        scale,
        bias, bool_trans_w=True, with_bias = True)

paddle.device.cuda.synchronize()
starttime = datetime.datetime.now()

for i in range(100):
    triton_output = triton_wint8(
        activation,
        no_perum_uint_qweight,
        scale,
        bias, 
        bool_trans_w = True, with_bias = True)

paddle.device.cuda.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("triton The whoel end to end time : ", time_ms, "ms")

no_perm_qweight = no_perm_qweight.transpose([1,0]).contiguous()

a = no_perm_qweight.astype("float16") * scale.reshape([1, N])
baseline = paddle.matmul(activation, a)
baseline += bias

#print("triton_output", triton_output)
#print("baseline", baseline)
print("triton and baseline diff", paddle.max(paddle.abs(triton_output - baseline)))
print("triton and paddle diff", paddle.max(paddle.abs(triton_output - paddle_output)))
