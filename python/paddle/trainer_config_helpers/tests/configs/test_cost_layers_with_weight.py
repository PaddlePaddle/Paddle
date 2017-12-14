from paddle.trainer_config_helpers import *

settings(learning_rate=1e-4, batch_size=1000)

data = data_layer(name='input', size=300)
lbl = data_layer(name='label', size=1)
wt = data_layer(name='weight', size=1)
fc = fc_layer(input=data, size=10, act=SoftmaxActivation())

outputs(
    classification_cost(
        input=fc, label=lbl, weight=wt),
    square_error_cost(
        input=fc, label=lbl, weight=wt),
    nce_layer(
        input=fc,
        label=data_layer(
            name='multi_class_label', size=500),
        weight=wt))
