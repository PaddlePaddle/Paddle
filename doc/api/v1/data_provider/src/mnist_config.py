from paddle.trainer_config_helpers import *

define_py_data_sources2(
    train_list='train.list',
    test_list=None,
    module='mnist_provider',
    obj='process')

img = data_layer(name='pixel', size=784)
label = data_layer(name='label', size=10)
