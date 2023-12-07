# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# from triton_ops import triton_fmha
# from triton_ops import triton_fmha2

# import paddle

# batch = 16
# seq_len = 512
# heads = 32
# head_dim = 128
# q = paddle.rand((batch, heads, seq_len, head_dim),dtype ="float16")
# k = paddle.rand((batch, heads, seq_len, head_dim),dtype ="float16")
# v = paddle.rand((batch, heads, seq_len, head_dim),dtype ="float16")
# seq_lens = paddle.to_tensor([[seq_len], [seq_len]]).astype("int32")
# scale=float(head_dim**-0.5)
# print(scale)


# attn_mask1 = paddle.full([batch, 1, seq_len,seq_len], -10000.0, 'float16')
# for i in range(seq_len):
#     for j in range(0, i + 1):
#         attn_mask1[:,:,i,j] = 0.0


# from paddle.incubate.nn.functional import (
#     variable_length_memory_efficient_attention,)

# import datetime

# starttime = datetime.datetime.now()
# for i in range(100):
#     #qkv_out1 = triton_fmha(q,k,v)
#     qkv_out2 = variable_length_memory_efficient_attention(q, k, v, seq_lens, seq_lens, mask=attn_mask1, scale=scale,)

# paddle.device.cuda.synchronize(0)
# endtime = datetime.datetime.now()
# duringtime = endtime - starttime
# time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
# print("The Origin Whole end to end time : ", time_ms, "ms")

# warm_up_times = 5
# repeat_times = 10

# # starttime = datetime.datetime.now()


# # for i in range(100):
# #     qkv_out1 = triton_fmha2(q,k,v)
# #     #qkv_out2 = variable_length_memory_efficient_attention(q, k, v, seq_lens, seq_lens, mask=attn_mask1, scale=scale,)

# # paddle.device.cuda.synchronize(0)
# # endtime = datetime.datetime.now()
# # duringtime = endtime - starttime
# # time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
# # print("The whole end to end time : ", time_ms, "ms")

# qkv_out1 = triton_fmha2(q,k,v)

# qkv_out3 = triton_fmha(q,k,v) 

# print(paddle.max(qkv_out2 - qkv_out1))

# print(paddle.max(qkv_out3 - qkv_out1))

# print(paddle.max(qkv_out3 - qkv_out2))


# print(qkv_out3)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from triton_ops import triton_fmha
from triton_ops import triton_fmha2

import paddle

batch = 16
seq_len = 512
heads = 32
head_dim = 128
q = paddle.rand((batch, heads, seq_len, head_dim),dtype ="float16")
k = paddle.rand((batch, heads, seq_len, head_dim),dtype ="float16")
v = paddle.rand((batch, heads, seq_len, head_dim),dtype ="float16")
seq_lens = paddle.to_tensor([[seq_len], [seq_len]]).astype("int32")
scale=float(head_dim**-0.5)





attn_mask1 = paddle.full([batch, 1, seq_len,seq_len], -10000.0, 'float16')
for i in range(seq_len):
    for j in range(0, i + 1):
        attn_mask1[:,:,i,j] = 0.0



from paddle.incubate.nn.functional import (
    variable_length_memory_efficient_attention,)




for i in range(100):
    qkv_out1 = triton_fmha(q,k,v)
    qkv_out2 = variable_length_memory_efficient_attention(q, k, v, seq_lens, seq_lens, mask=attn_mask1, scale=scale,)

paddle.device.cuda.synchronize(0)



import datetime

warm_up_times = 5
repeat_times = 10

starttime = datetime.datetime.now()


for i in range(100):
    qkv_out1 = triton_fmha(q,k,v)
    #qkv_out2 = variable_length_memory_efficient_attention(q, k, v, seq_lens, seq_lens, mask=attn_mask1, scale=scale,)

paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The whoel end to end time : ", time_ms, "ms")

qkv_out3 = triton_fmha2(q,k,v)

print(paddle.max(qkv_out2 - qkv_out1))
#print(paddle.max(qkv_out3 - qkv_out1))
#print(paddle.max(qkv_out3 - qkv_out2))


print((qkv_out3 - qkv_out2)[1][1][0])



