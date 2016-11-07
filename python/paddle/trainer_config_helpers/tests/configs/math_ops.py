from paddle.trainer_config_helpers import *
from paddle.trainer_config_helpers import math

settings(
    batch_size=1000,
    learning_rate=1e-5
)

x = data_layer(name='data', size=100)
x = math.exp(x)
x = math.log(x)
x = math.abs(x)
x = math.sigmoid(x)
x = math.square(x)
x = math.square(x)
y = 1 + x
y = y + 1
y = x + y
y = y - x
y = y - 2
y = 2 - y

outputs(y)

