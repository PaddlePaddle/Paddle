from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=300)
resized = resize_layer(input=data, size=150)

outputs(resized)
