from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

weight = data_layer(name='w', size=1)
a = data_layer(name='a', size=100)
b = data_layer(name='b', size=100)
c = data_layer(name='c', size=200)
d = data_layer(name='d', size=31)

outputs(
    interpolation_layer(
        input=[a, b], weight=weight),
    power_layer(
        input=a, weight=weight),
    scaling_layer(
        input=a, weight=weight),
    cos_sim(
        a=a, b=b),
    cos_sim(
        a=a, b=c, size=2),
    sum_to_one_norm_layer(input=a),
    conv_shift_layer(
        a=a, b=d),
    tensor_layer(
        a=a, b=b, size=1000),
    slope_intercept_layer(
        input=a, slope=0.7, intercept=0.9),
    linear_comb_layer(
        weights=b, vectors=c))
