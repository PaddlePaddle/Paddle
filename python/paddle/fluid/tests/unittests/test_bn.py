import paddle
import torch
import time

shape=[126000, 16]
x = paddle.randn(shape)
print(x.shape)

bn = paddle.nn.BatchNorm1D(16)

#warm up
out = bn(x)
paddle.device.cuda.synchronize()
print(out.shape)

t0 = time.time()
for i in range(100):
    out = bn(x)
paddle.device.cuda.synchronize()
t1 = time.time()
print("paddle time : ", t1-t0)


device = torch.device("cuda")
torch_x = torch.tensor(x.numpy(), device=device)

torch_bn = torch.nn.BatchNorm1d(16, device=device)

print(torch_x.shape)
torch_out = torch_bn(torch_x)
torch.cuda.synchronize(device)
print(torch_out.shape)

t0 = time.time()
for i in range(100):
    torch_out = torch_bn(torch_x)
torch.cuda.synchronize(device)
t1 = time.time()
print("torch time : ", t1-t0)
