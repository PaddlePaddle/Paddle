from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=300, height=100, width=3)

resize = resize_layer(input=data, size=100)

outputs(resize)