'''
Test all activations.
'''

from paddle.trainer_config_helpers import *

settings(learning_rate=1e-4, batch_size=1000)

din = data_layer(name='input', size=100)

acts = [
    TanhActivation, SigmoidActivation, SoftmaxActivation, IdentityActivation,
    LinearActivation, ExpActivation, ReluActivation, BReluActivation,
    SoftReluActivation, STanhActivation, AbsActivation, SquareActivation
]

outputs([
    fc_layer(
        input=din, size=100, act=act(), name="layer_%d" % i)
    for i, act in enumerate(acts)
])
