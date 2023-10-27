import torch 
import numpy as np

x_np = np.load('hs_0.npy')
x = torch.tensor(x_np, dtype=torch.bfloat16)
w_np = np.load('w.npy')
w = torch.tensor(w_np, dtype=torch.bfloat16)

out = torch.matmul(x, w.transpose(0,1))
print(out)
np.save('t_sdpa.npy', out.to(torch.float32))
