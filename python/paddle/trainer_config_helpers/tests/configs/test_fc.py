from paddle.trainer_config_helpers import *

"""
For testing optimizer with momentum setting

1. momentum and sparse can not be set simutanously except MomentumOptimizer
settings(batch_size=1000,
         learning_method=AdaGradOptimizer(momentum=0.9, sparse=True),
         learning_rate=1e-5)

2. default_momentum and momentum attribution should be set exclusively
   for better understanding.
default_momentum(0.4)
settings(batch_size=1000,
         learning_method=AdaGradOptimizer(momentum=0.9, sparse=True),
         learning_rate=1e-5)

In theory any optimizer with momentum setting do not work with sparse upate
except MomentumOptimizer which has implemented specical algorithm.
"""

settings(batch_size=1000,
         learning_method=AdaGradOptimizer(momentum=0.9),
         learning_rate=1e-5)

din = data_layer(name='data', size=100)

trans = trans_layer(input=din)

hidden = fc_layer(input=trans, size=100, bias_attr=False)

mask = data_layer(name='mask', size=100)

hidden_sel = selective_fc_layer(
    input=din, select=mask, size=100, act=SigmoidActivation())

outputs(hidden, hidden_sel)
