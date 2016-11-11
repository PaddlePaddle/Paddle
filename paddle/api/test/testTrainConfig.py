from paddle.trainer_config_helpers import *

settings(batch_size=100, learning_method=AdamOptimizer())

din = data_layer(name='input', size=784)

fc1 = fc_layer(name='hidden1', input=din, size=100)
fc2 = fc_layer(name='hidden2', input=fc1, size=100)

opt = fc_layer(input=fc2, size=10, act=SoftmaxActivation())
outputs(classification_cost(input=opt, label=data_layer('lbl', 10)))
