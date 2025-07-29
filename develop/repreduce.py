import paddle
import numpy as np

import torch

def paddle_set_item():
    input_shape = [2, 3, 3, 3]
    index = paddle.to_tensor(np.ones([3, 3], dtype="bool"))
    index[0][0]=0
    value = -1
    input_tensor = paddle.arange(np.prod(input_shape), dtype="int").reshape(input_shape)
    input_tensor[0,:,index] = value
    return input_tensor
    
def torch_set_item():
    input_shape = [2, 3, 3, 3]
    index = torch.ones([3, 3], dtype=torch.bool)
    index[0][0]=0
    value = -1
    input_tensor = torch.arange(np.prod(input_shape), dtype=torch.int32).reshape(input_shape)
    input_tensor[0,:,index] = value
    return input_tensor

def paddle_error_case():
    input_shape = [3, 4, 5, 6, 7]
    np_tensor = np.arange(np.prod(input_shape), dtype=np.int32).reshape(input_shape)
    input_tensor = paddle.arange(np.prod(input_shape), dtype="int32").reshape(input_shape)
    out_np = np_tensor[::2, [1, 0], [2, 3], 0:4:2]
    print("Numpy output shape:", out_np.shape)
    print("Numpy output:", out_np)
    
    out_paddle = input_tensor[::2, [1, 0], [2, 3], 0:4:2]
    print("Paddle output shape:", out_paddle.shape)
    print("Paddle output:", out_paddle.numpy())
    np.testing.assert_array_equal(out_paddle.numpy(), out_np)
    return
    
def paddle_get_item():
    input_shape = [2, 3, 3, 3]
    index = paddle.to_tensor(np.ones([3, 3], dtype="bool"))
    index[0][0]=0
    input_tensor = paddle.arange(np.prod(input_shape), dtype="int").reshape(input_shape)
    out = input_tensor[0,:,index]
    print("Paddle output shape:", out.shape)
    print("Paddle output:", out.numpy())
    return out

def torch_get_item():
    input_shape = [2, 3, 3, 3]
    index = torch.ones([3, 3], dtype=torch.bool)
    index[0][0]=0
    input_tensor = torch.arange(np.prod(input_shape), dtype=torch.int32).reshape(input_shape)
    out = input_tensor[0,:,index]
    print("Torch output shape:", out.shape)
    print("Torch output:", out.numpy())
    return out

def paddle_empty_case2():
    x = paddle.randn((108, 64, 12288))
    index = np.ones((108), dtype=bool), slice(None, None, None), -1
    paddle.device.cuda.synchronize()
    for _ in range(1000):
        y = x[index].shape
    print(y)
    
def test_paddle_sum():
    x = paddle.to_tensor([True])
    y = paddle.sum(x)
    print(y)

if __name__ == "__main__":
    # test_paddle_sum()
    # torch_get_item()
    # paddle_get_item()
    paddle_empty_case2()