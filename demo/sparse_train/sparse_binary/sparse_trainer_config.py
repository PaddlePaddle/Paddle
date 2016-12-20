from paddle.trainer_config_helpers import *

label_size = 2
data_size = 18182296


""" Algorithm Configuration """
settings(learning_rate=1e-3,
         learning_method=MomentumOptimizer(momentum=0.9, sparse=True),
         batch_size=200)

""" Data Configuration """
define_py_data_sources2(train_list='train.list',
                        test_list=None,
                        module='sparse_data_provider',
                        obj='process')

""" Model Configuration """
data= data_layer(name='data',
                       size=data_size)
label = data_layer(name='label',
                   size=label_size)

hidden1 = fc_layer(input=data,
                   size=32,
                   param_attr=ParameterAttribute(sparse_update=True))
hidden2 = fc_layer(input=hidden1,
                   size=32)

prediction = fc_layer(input=hidden2, size=label_size, act=SoftmaxActivation())

outputs(classification_cost(input=prediction, label=label))
