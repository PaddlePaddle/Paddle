from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=300, height=10, width=10)
prelu = prelu_layer(input=data, num_channels=3)
prelu = prelu_layer(input=data, partial_sum=1, num_channels=3)
prelu = prelu_layer(input=data, partial_sum=5, num_channels=3)
prelu = prelu_layer(input=data, channel_shared=True, num_channels=3)
prelu = prelu_layer(input=data, channel_shared=False, num_channels=3)

outputs(prelu)
