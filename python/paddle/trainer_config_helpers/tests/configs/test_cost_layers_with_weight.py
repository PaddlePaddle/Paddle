from paddle.trainer_config_helpers import *

settings(learning_rate=1e-4, batch_size=1000)

data = data_layer(name='input', size=300)
lbl = data_layer(name='label', size=1)
wt = data_layer(name='weight', size=1)
fc = fc_layer(input=data, size=10, act=SoftmaxActivation())

outputs(
    classification_cost(
        input=fc, label=lbl, weight=wt),
    mse_cost(
        input=fc, label=lbl, weight=wt))
