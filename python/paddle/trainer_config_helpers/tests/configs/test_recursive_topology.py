from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

din = data_layer(name='data', size=100)

enc = din
for i in range(32):
    enc = addto_layer([enc, enc])

pred = fc_layer(
    input=fc_layer(
        input=enc, size=32, act=ReluActivation()),
    size=10,
    act=SoftmaxActivation())
outputs(pred)
