from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

data = data_layer(name='data', size=1024)

fm = factorization_machine(input=data, factor_size=10)

outputs(fm)
