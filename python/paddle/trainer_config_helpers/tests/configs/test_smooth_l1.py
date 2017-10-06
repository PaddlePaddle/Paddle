from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=300)
lbl = data_layer(name='label', size=300)
smooth_l1 = smooth_l1_cost(input=data, label=lbl)

outputs(smooth_l1)
