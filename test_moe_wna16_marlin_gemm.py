from paddle.base.core import moe_wna16_marlin_gemm
import paddle
# from paddlenlp_ops import (
#     fused_expert_moe,
#     moe_expert_dispatch,
#     moe_expert_ffn,
#     moe_expert_reduce,
#     moe_gemm
# )
from paddle.nn.quant import weight_quantize

# import triton
# import triton.language as tl
import time
       
def GetQuantizedWeights(num_expert, bmm_w0, quant_method, group_size=-1):
    """
    Quantizes the weights for the experts' layers and returns the quantized weights and scales.
    :param quant_method: The quantization method to use.
    :return: Quantized bmm_w0, bmm_w1, scale0, scale1
    """
    a,b,c = bmm_w0.shape
    d_model = b
    if quant_method != "None":
        fc0_expert_weights_for_ref_list = []
        scale0 = []
        for i in range(num_expert):
            fc0_expert_weights_for_ref_i, fc0_expert_weights_scale_for_ref_i = weight_quantize(bmm_w0[i], algo=quant_method, group_size=group_size)

            fc0_expert_weights_for_ref_list.append(
                fc0_expert_weights_for_ref_i.reshape(
                    [d_model, -1]
                    if quant_method == "weight_only_int8"
                    else [d_model, -1]
                )
            )
            scale0.append(fc0_expert_weights_scale_for_ref_i)

        
        bmm_w0_quantized = paddle.to_tensor(fc0_expert_weights_for_ref_list)
        scale0 = paddle.to_tensor(scale0)
        
        return bmm_w0_quantized, scale0

def run_paddle_moe(inputs, weight, scores, topk, bit):
    if bit == 4:
        quant_mode = "weight_only_int4"
    else:
        quant_mode = "weight_only_int8"
    print(quant_mode)
    group_size = 64
    expert = scores.shape[1]
    
    # (
    #     inputs,
    #     token_nums_per_expert,
    #     permute_indices_per_token,
    #     top_k_weights,
    #     top_k_indices,
    # ) = moe_expert_dispatch(inputs, scores, topk, False, topk_only_mode=True)
    w1, scale1 = GetQuantizedWeights(expert , weight, quant_mode, -1)
    
    print("w1.shape: ", w1.shape)
    
    sorted_token_ids     = paddle.zeros([num_tokens * topk], dtype='int64')  # 全部指向专家 0
    expert_ids           = paddle.zeros([num_tokens * topk], dtype='int64')  # 全部专家 ID=0
    num_tokens_past_padded = paddle.to_tensor([num_tokens], dtype='int64')                                  
    topk_weights         = paddle.ones([num_tokens, topk], dtype='float32')
    
    c_or_none            = None
    global_scale_or_none = None
    b_zeros_or_none      = None
    g_idx_or_none        = None
    perm_or_none         = None

    workspace = paddle.empty(
        [num_tokens * topk, k],
        dtype=inputs.dtype,
    )

    # counts            = token_nums_per_expert
    # num_tokens_padded = ((counts + group_size - 1) // group_size) * group_size
    # num_tokens_padded = num_tokens

    mul_topk_weights  = False
    is_ep             = False
    b_q_type_id       = 4   # 或者 8
    use_atomic_add    = False
    use_fp32_reduce   = False
    is_zp_float       = False
    
    out = moe_wna16_marlin_gemm(
        inputs,        # a
        c_or_none,
        w1,                   # b_q_weight
        scale1,               # b_scales
        global_scale_or_none,
        b_zeros_or_none,
        g_idx_or_none,
        perm_or_none,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        topk_weights,
        group_size,           # moe_block_size
        topk,                 # top_k
        mul_topk_weights,
        is_ep,
        b_q_type_id,
        num_tokens,           # size_m
        n,                    # size_n
        k,                    # size_k
        True,                 # is_k_full
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float
    )
    
    def run():
        out = moe_gemm(
                    inputs,
                    token_nums_per_expert,
                    w1,
                    None,
                    scale1,
                    quant_mode
                )
        # print(out)
        return out

    # with paddle.device.cuda.stream_guard(paddle.device.cuda.Stream()):
    #     paddle.device.cuda.synchronize()
    #     t = triton.testing.do_bench_cudagraph(run, rep=100)
    #     paddle.device.cuda.synchronize()
    paddle.device.cuda.synchronize()  # 确保初始同步
    start = time.time()
    # for _ in range(100):
    #     run()
    paddle.device.cuda.synchronize()  # 确保计算完成
    t = (time.time() - start) / 100 * 1000 * 1e3  # 平均时间（秒）

    return t



weight_shapes = [
    (256, 8, 2048, 8192),
    # (256, 8, 7168, 256),

    # (60, 4, 1408 * 2, 2048),
    # (60, 4, 2048, 1408),
]

res = []
for num_experts, top_k, n, k in weight_shapes:
    # for bit in [4, 8]:
    for bit in [4,]:
        for i in range(4):
            num_tokens = 2 ** i

            paddle.seed(0)

            inputs = paddle.randn(
                (num_tokens, k),
                dtype="bfloat16"
            ) / 10
            weight = paddle.randn(
                (num_experts, n, k),
                dtype="bfloat16"
            ) / 10

            scores = paddle.randn( 
                (num_tokens, num_experts),
                dtype="float32"
            )

            res.append({
                "num_experts": num_experts,
                "top_k": top_k,
                "n": n,
                "k": k,
                "bit": bit,
                "num_tokens": num_tokens,
                "paddle":run_paddle_moe(inputs, weight, scores, top_k, bit),
            })
print(res)