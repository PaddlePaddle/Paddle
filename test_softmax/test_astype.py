import paddle
import numpy as np
np.random.seed(10)

shape = [2, 3, 4]
np_x = np.random.random(size=shape).astype("float32")
x_1 = paddle.to_tensor(np_x, dtype = "bfloat16")
x_2 = paddle.to_tensor(np_x).astype("bfloat16")

print(paddle.abs(x_1 - x_2))