import paddle
from paddle.nn.quant import weight_quantize
from paddle.nn.quant import weight_only_linear

from triton_ops import triton_wint4

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

M = 16
N = 4096*3
K = 4096

activation = paddle.randn((M, K), dtype=paddle.float16)
original_weight = paddle.randn((K, N), dtype=paddle.float16)

paddle.set_device("cpu")
perm_qweight, scale = weight_quantize(original_weight.cpu(), algo="weight_only_int4")
paddle.set_device("gpu")
perm_qweight = perm_qweight.cuda()
scale = scale.cuda()

bias = paddle.rand((N,), dtype=paddle.float16) * 10

no_perm_qweight = original_weight / scale.reshape([1, N])
no_perm_qweight = paddle.round(no_perm_qweight)
no_perm_qweight = paddle.clip(no_perm_qweight, min=-7, max=7)
no_perum_uint_qweight = no_perm_qweight.astype("int32") + 8
no_perum_uint_qweight = no_perum_uint_qweight.astype("uint8")


# tmp = paddle.assign(no_perum_uint_qweight)
# tmp = tmp.reshape([K // 64, 64,  N])
# tmp0 = paddle.assign(tmp[:, 0:64//2, :])
# tmp1 = paddle.assign(tmp[:, 64//2:64, :])
# tmp[:,0::2,:] = tmp0
# tmp[:,1::2,:] = tmp1
# no_perum_uint_qweight = tmp.reshape([K, N])


a = no_perum_uint_qweight[0::2,:]
b = no_perum_uint_qweight[1::2,:] * 16
a = a & paddle.to_tensor([0b1111],dtype="uint8")
b = b & paddle.to_tensor([0b11110000],dtype="uint8")

pack_no_perm_uint_qweight = a | b

paddle.device.cuda.synchronize()

# 下面是paddle的cutlass代码
import datetime
for i in range(100):
    paddle_output = weight_only_linear(activation, perm_qweight, bias, scale, weight_dtype = "int4")

paddle.device.cuda.synchronize()
starttime = datetime.datetime.now()

for i in range(100):
    paddle_output = weight_only_linear(activation, perm_qweight, bias, scale, weight_dtype = "int4")

paddle.device.cuda.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("paddle cutlass The whoel end to end time : ", time_ms, "ms")



# 下面是triton的计算代码

assert activation.is_contiguous()
assert pack_no_perm_uint_qweight.is_contiguous()
assert scale.is_contiguous()

for i in range(100):
    triton_output = triton_wint4(
        activation,
        pack_no_perm_uint_qweight,
        scale,
        bias, bool_trans_w=False, with_bias = True)

paddle.device.cuda.synchronize()
starttime = datetime.datetime.now()

for i in range(100):
    triton_output = triton_wint4(
        activation,
        pack_no_perm_uint_qweight,
        scale,
        bias, 
        bool_trans_w = False, with_bias = True)

paddle.device.cuda.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("triton The whoel end to end time : ", time_ms, "ms")

a = (no_perm_qweight.astype("float16")) * scale.reshape([1, N])
baseline = paddle.matmul(activation, a)
baseline += bias

#print("triton_output", triton_output)
print("baseline and paddle_output diff", paddle.max(paddle.abs(paddle_output - baseline)))

print("triton and baseline diff", paddle.max(paddle.abs(triton_output - baseline)))
print("triton and paddle diff", paddle.max(paddle.abs(triton_output - paddle_output)))