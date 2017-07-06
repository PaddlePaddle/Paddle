from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=300)
prelu = prelu_layer(input=data)

outputs(prelu)
