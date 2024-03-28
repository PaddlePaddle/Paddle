from numpy import bool_
import paddle
from triton_ops import triton_w4a8

M = 16
N = 4096
K = 4096
activation = (paddle.randn((M, K), dtype=paddle.float32) * 100).astype("int8")


# 下面是triton的计算代码

qweight = (paddle.randn((K // 8, N), dtype=paddle.float32) * 100 - 2).astype("int32")
#qweight = (paddle.randn((K // 2, N), dtype=paddle.float32) * 100 - 2).astype("int8")
ele_per_btype = 8

bool_trans_w = False

if bool_trans_w:
    qweight = qweight.transpose([1, 0])

import datetime
for i in range(100):
    triton_output = triton_w4a8(activation, qweight, bool_trans_w=bool_trans_w)

paddle.device.cuda.synchronize()
starttime = datetime.datetime.now()

for i in range(100):
    triton_output = triton_w4a8(activation, qweight, bool_trans_w=bool_trans_w)

paddle.device.cuda.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("triton The whoel end to end time : ", time_ms, "ms")


# baseline computed by paddle op.

if bool_trans_w:
    qweight = qweight.transpose([1, 0])
qweight = qweight.numpy()
unpack_qweight = paddle.zeros((K, N), dtype=paddle.int8).numpy()
for i in range(K):
    for j in range(N):
        int4_id = i % ele_per_btype
        int32 = qweight[i // ele_per_btype, j]
        int32 = (int32 >> (int4_id * 4) << 4) & 0b11110000
        unpack_qweight[i, j] = int32

unpack_qweight = paddle.to_tensor(unpack_qweight)
unpack_qweight = unpack_qweight.transpose([1, 0])


#activation = (paddle.randn((M, K), dtype=paddle.float32) * 100).astype("float16")
#unpack_qweight = (paddle.randn((K, N), dtype=paddle.float32) * 100).astype("float16")

for i in range(100):
    paddle_out = paddle.matmul(activation, unpack_qweight, False, True)

paddle.device.cuda.synchronize()
starttime = datetime.datetime.now()

for i in range(100):
    paddle_out = paddle.matmul(activation, unpack_qweight, False, True)

paddle.device.cuda.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("paddle The whoel end to end time : ", time_ms, "ms")


print("paddle_out", paddle.max(paddle.abs(paddle_out - triton_output)))
