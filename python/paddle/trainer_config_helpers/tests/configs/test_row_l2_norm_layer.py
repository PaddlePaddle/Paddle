from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=300)
row_l2_norm = row_l2_norm_layer(input=data)

outputs(row_l2_norm)
