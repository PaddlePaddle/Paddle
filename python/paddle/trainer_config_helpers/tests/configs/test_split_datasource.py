from paddle.trainer_config_helpers import *

setup_data_provider("train.list", "test.list", ["a", "b"], ("c", "d"), None)
settings(learning_rate=1e-3, batch_size=1000)

outputs(data_layer(name="a", size=10))
