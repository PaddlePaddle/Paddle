from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=300)
prelu = prelu_layer(input=data)
prelu = prelu_layer(input=data, partial_sum=1)
prelu = prelu_layer(input=data, partial_sum=5)

outputs(prelu)
