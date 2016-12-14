from paddle.trainer_config_helpers import *

define_py_data_sources2(
    train_list=["do_not_matter.txt"],
    test_list=None,
    module='sparse_updated_network_provider',
    obj='process')

settings(batch_size=100, learning_rate=1e-4)

outputs(
    classification_cost(
        input=fc_layer(
            size=10,
            act=SoftmaxActivation(),
            input=embedding_layer(
                size=64,
                input=data_layer(
                    name='word_id', size=600000),
                param_attr=ParamAttr(sparse_update=True))),
        label=data_layer(
            name='label', size=10)))
