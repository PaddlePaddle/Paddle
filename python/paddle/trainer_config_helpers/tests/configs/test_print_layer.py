from paddle.trainer_config_helpers import *

settings(learning_rate=1e-4, batch_size=1000)

din = data_layer(name='input', size=100)

print_layer(input=din)

outputs(din)
