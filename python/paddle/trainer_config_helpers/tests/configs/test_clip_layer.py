from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=300)
clip = clip_layer(input=data, min=-10, max=10)

outputs(clip)
