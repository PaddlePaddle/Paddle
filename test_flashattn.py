import paddle
import paddle.nn.functional as F
import numpy as np
import time
def time_paddle():
    paddle.device.synchronize()
    return time.time()


def time_it(run_iter: int, warmup_iter: int, target_func):
    for idx in range(run_iter + warmup_iter):
        if idx == warmup_iter:
            sta_time = time_paddle()
        target_func()
    end_time = time_paddle()
    return (end_time - sta_time) / run_iter

def genqkv(bsz, sq, sk, hq, hk, hdim, dtype):
    shape_q = (bsz, sq, hq, hdim)
    shape_k = (bsz, sk, hk, hdim)
    q = paddle.randn(shape_q, dtype=dtype)
    k = paddle.randn(shape_k, dtype=dtype)
    v = paddle.randn(shape_k, dtype=dtype)
    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False
    return q, k, v

if __name__ == "__main__":
    bsz = 1
    sq = 8192
    sk = 8192
    hq = 8
    hk = 8
    hdim = 128
    dtype = "float16"
    paddle.seed(0)
    q, k, v = genqkv(bsz, sq, sk, hq, hk, hdim, dtype)

    def fa_func():
        o = F.flash_attention.flash_attention(q, k, v, causal=True)
        o[0].backward()
        return o[0]
    x = fa_func()
    np.save("out1.npy",x.numpy())
    t = time_it(100, 10, fa_func)
    print(f"FlashAttention Time: {t}s")
