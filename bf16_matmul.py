import paddle
import paddle.nn.functional as F
import numpy as np

x_np = np.load('hs_0.npy')
x = paddle.to_tensor(x_np, dtype='bfloat16')
w_np = np.load('w.npy')
w = paddle.to_tensor(w_np, dtype='bfloat16')

out = F.linear(x, w.T, bias=None)
out1 = paddle.matmul(x, w, transpose_y=True)
print(out)
print(paddle.all((out-out1)==0))
np.save('sdpa.npy', out.astype('float32'))
np.save('sdpa1.npy', out1.astype('float32'))
import pdb; pdb.set_trace()
