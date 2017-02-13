from paddle.trainer_config_helpers import *

setup_data_provider('train.list', None, 'mnist_provider', 'process')

img = data_layer(name='pixel', size=784)
label = data_layer(name='label', size=10)
