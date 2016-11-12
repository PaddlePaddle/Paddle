from paddle.trainer_config_helpers import *

define_py_data_sources2(
    train_list="train.list",
    test_list="test.list",
    module=["a", "b"],
    obj=("c", "d"))
settings(learning_rate=1e-3, batch_size=1000)

outputs(data_layer(name="a", size=10))
