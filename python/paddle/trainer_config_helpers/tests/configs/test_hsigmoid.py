from paddle.trainer_config_helpers import *

settings(learning_rate=1e-4, batch_size=1000)

din = data_layer(name='data', size=100)
label = data_layer(name='label', size=10)

outputs(hsigmoid(input=din, label=label, num_classes=10))
