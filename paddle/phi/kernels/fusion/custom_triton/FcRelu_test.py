import paddle
from triton_ops import triton_FcRelu
M = 900
N = 256
K = 256
a = paddle.rand((M, K),dtype ="float16")
b = paddle.rand((K, N),dtype ="float16")
bias = paddle.rand((K, ),dtype ="float16")
triton_c = triton_FcRelu(a, b, bias)

c = paddle.matmul(a, b)
c = c + bias
m = paddle.nn.ReLU()
out = m(c)

print(paddle.max(c - triton_c))

import datetime
repeat_times = 100
warm_up_times = 20
# paddle op
for i in range(warm_up_times):
    c = paddle.matmul(a, b)
    c = c + bias
    m = paddle.nn.ReLU()
    out = m(c)
paddle.device.cuda.synchronize(0)

starttime = datetime.datetime.now()
for i in range(repeat_times):
    c = paddle.matmul(a, b)
    c = c + bias
    m = paddle.nn.ReLU()
    out = m(c)
paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The PADDLE OP end to end time : ", time_ms, "ms")

# triton op
for i in range(warm_up_times):
    triton_c = triton_FcRelu(a,b,bias)
paddle.device.cuda.synchronize(0)   
starttime = datetime.datetime.now()
for i in range(repeat_times):
    triton_c = triton_FcRelu(a,b,bias)
paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The TRITON OP end to end time : ", time_ms, "ms")