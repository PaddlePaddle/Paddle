from paddle.trainer_config_helpers import *

label_size = 1
data_size = 10000000


""" Algorithm Configuration """
settings(learning_rate=1e-3,
         learning_method=MomentumOptimizer(momentum=0.9),
         batch_size=200)

""" Data Configuration """
define_py_data_sources2(train_list='train.list',
                        test_list=None,
                        module='sparse_float_data_provider',
                        obj='process')

""" Model Configuration """
sparse_float = data_layer(name='data',
                       size=data_size)
label = data_layer(name='label',
                   size=label_size)

hidden1 = fc_layer(input=sparse_float,
                   size=512,
                   param_attr=ParameterAttribute(sparse_update=True))
hidden2 = fc_layer(input=hidden1,
                   size=1)
outputs(regression_cost(input=hidden2, label=label))
