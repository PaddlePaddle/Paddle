from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

x = data_layer(name='data', size=100)
x = layer_math.exp(x)
x = layer_math.log(x)
x = layer_math.abs(x)
x = layer_math.sigmoid(x)
x = layer_math.tanh(x)
x = layer_math.square(x)
x = layer_math.relu(x)
y = 1 + x
y = y + 1
y = x + y
y = y - x
y = y - 2
y = 2 - y
y = 2 * y
y = y * 3
z = data_layer(name='data_2', size=1)
y = y * z
y = z * y
y = y + z
y = z + y
outputs(y)
