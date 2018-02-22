from paddle.trainer_config_helpers import *

data = data_layer(name='data', size=1024)

fm = factorization_machine(input=data, factor_size=10)

outputs(fm)
