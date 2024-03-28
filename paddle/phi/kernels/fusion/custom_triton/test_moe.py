
import paddle
paddle.seed(123)
from triton_ops import triton_moe

import numpy as np

# npzfile = np.load('/zhoukangkang/outfile.npz')
# hidden_states = paddle.to_tensor(npzfile['hidden_states'])
# all_experts_weight1 = paddle.to_tensor(npzfile['all_experts_weight1'])
# routing_weights = paddle.to_tensor(npzfile['routing_weights'])
# selected_experts = paddle.to_tensor(npzfile['selected_experts'])

fake_top_k = 2 # this is used to 
M = 32
K = 4096
num_expert = 8
N = 3584
top_k = 2

hidden_states = paddle.randn((M, K), dtype=paddle.float16)
all_experts_weight1 = paddle.randn((num_expert, K, N), dtype=paddle.float16)
routing_weights = paddle.randn((M, num_expert), dtype=paddle.float16)
routing_weights, selected_experts = paddle.topk(routing_weights, top_k, axis=-1)
selected_experts = selected_experts.astype("int32")

print("selected_experts", selected_experts)

for i in range(50):
    c_out = triton_moe(hidden_states, all_experts_weight1, routing_weights, selected_experts, fake_top_k)


paddle.device.cuda.synchronize()
import datetime
starttime = datetime.datetime.now()

for i in range(100):
    c_out = triton_moe(hidden_states, all_experts_weight1, routing_weights, selected_experts, fake_top_k)

paddle.device.cuda.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The triton_moe end to end time : ", time_ms, "ms")

baseline_cache = paddle.zeros((M, top_k, N), dtype=hidden_states.dtype)

def baseline():
    for expert_id in range(num_expert):
        token_id, token_topkid_in_this_expert = paddle.where(selected_experts == expert_id)
        this_num = int(token_id.numel())
        token_id = token_id.reshape([this_num])
        token_topkid_in_this_expert = token_topkid_in_this_expert.reshape([this_num])
        current_state = paddle.gather(hidden_states, token_id)
        tmp = paddle.matmul(current_state, all_experts_weight1[expert_id])
        baseline_cache[token_id,token_topkid_in_this_expert,:] = tmp


baseline_cache = paddle.zeros((M, top_k, N), dtype=hidden_states.dtype)
for i in range(50):
    baseline()

paddle.device.cuda.synchronize()
import datetime
starttime = datetime.datetime.now()

for i in range(100):
    baseline()

paddle.device.cuda.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The baseline end to end time : ", time_ms, "ms")

print(paddle.max(paddle.abs(baseline_cache - c_out)))

