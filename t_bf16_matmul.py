import torch 
import torch.nn.functional as F
import numpy as np

x_np = np.load('hs_0.npy')
x = torch.tensor(x_np, dtype=torch.bfloat16).to("cuda:0")
w_np = np.load('w.npy')
w = torch.tensor(w_np, dtype=torch.bfloat16).to("cuda:0")

out1 = torch.matmul(x, w.transpose(0,1))
out = F.linear(x, w, bias=None)
print(out)
print(torch.all((out1-out) == 0))
np.save('t_sdpa.npy', out.cpu().detach().to(torch.float32))
np.save('t_sdpa1.npy', out1.cpu().detach().to(torch.float32))
