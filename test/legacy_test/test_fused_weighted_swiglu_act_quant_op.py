import numpy as np
import paddle
import paddle.incubate.nn.functional as F
#import test_quant

'''
Swiglu Function:
out = silu(x) * y when y is not None
out = silu(xs[0]) * xs[1] when y is None, where xs = paddle.chunk(x, 2, axis=-1)
'''

def dequantize_fp8_to_bf16(fp8_tensor: paddle.Tensor, 
                           scale: paddle.Tensor) -> paddle.Tensor:
    expanded_scale = paddle.repeat_interleave(
        scale, 
        repeats=128, 
        axis=-1
    )
    # 非规整情况，需要截断
    expanded_scale = expanded_scale[:, :fp8_tensor.shape[-1]]
    return (fp8_tensor.astype('float32') * expanded_scale)

def printany(te):
    for i in range(te.shape[0]):
        for j in range(te.shape[1]):
            print(te[i][j], end=", ")
        print()
    print("-"*20)

def verify():
    for width in [4096,7168]:
        for height in [8192, 16384, 32768]:
            print("#"*60 + f" Testing width:{width}, height:{height} " + "#"*60)
            x= paddle.clip(paddle.randn([height, width]).astype("bfloat16"), min=-50, max=50)
            prob = paddle.randn([height, 1]).astype("float32")
            np_results=[]
            golden_res = F.swiglu(x) * prob 
            fused_res, fused_scales = paddle.nn.functional.fused_weighted_swiglu_act_quant(x,prob, using_pow2_scaling=False)
            np_results.append(golden_res.astype("float").numpy())
            np_results.append(dequantize_fp8_to_bf16(fused_res, fused_scales).numpy())
            nan_cnt_golden, nan_cnt_fused= np.sum(np.isnan(np_results[0])), np.sum(np.isnan(np_results[1]))
            print(f"Nan count of Golden result: {nan_cnt_golden}; Nan count of Fused result: {nan_cnt_fused}")
            try:
                np.testing.assert_allclose(np_results[0], np_results[1], rtol=0.01, atol=1) #存在截断误差，atol=1，通常在1e-6
                print("+++++++ Passed ++++++++")
            except AssertionError as err:
                print(err)
            print(np_results[0])
            print("_________")
            print(np_results[1])
                #compare_tensors(np_results[0], np_results[1])

def run():
    verify()

if __name__ == "__main__":
    run()