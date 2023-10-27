import paddle
import numpy as np

x_np = np.load('hs_0.npy')
x = paddle.to_tensor(x_np, dtype='bfloat16')
w_np = np.load('w.npy')
w = paddle.to_tensor(w_np, dtype='bfloat16')

out = paddle.matmul(x, w, transpose_y=True)
print(out)
np.save('sdpa.npy', out.astype('float32'))
