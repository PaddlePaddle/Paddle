import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from triton_ops import triton_matmul

import paddle

a = paddle.rand([4096,4096],dtype="float16")
b = paddle.rand([4096,4096],dtype="float16")
c1 = triton_matmul(a,b)
c2 = paddle.matmul(a,b)

for i in range(100):
    c1 = triton_matmul(a,b)
    c2 = paddle.matmul(a,b)


paddle.device.cuda.synchronize(0)

import datetime
starttime = datetime.datetime.now()

for i in range(100):
    c1 = triton_matmul(a,b)
    #c2 = paddle.matmul(a,b)
paddle.device.cuda.synchronize(0)

endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The whoel end to end time : ", time_ms, "ms")


print(c1-c2)
