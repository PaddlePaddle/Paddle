from paddle.trainer_config_helpers import *

settings(batch_size=128, learning_method=AdaGradOptimizer(), learning_rate=1e-4)

file_list = 'trainer/tests/fake_file_list.list'

define_py_data_sources2(
    train_list=file_list,
    test_list=file_list,
    module="simple_sparse_neural_network_dp",
    obj="process")

embedding = embedding_layer(
    input=data_layer(
        name="word_ids", size=65536),
    size=128,
    param_attr=ParamAttr(sparse_update=True))
prediction = fc_layer(input=embedding, size=10, act=SoftmaxActivation())

outputs(
    classification_cost(
        input=prediction, label=data_layer(
            name='label', size=10)))
