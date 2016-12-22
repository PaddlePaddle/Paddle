from paddle.trainer_config_helpers import *

settings(
    learning_rate=1e-4,
    learning_method=AdamOptimizer(),
    batch_size=1000,
    model_average=ModelAverage(average_window=0.5),
    regularization=L2Regularization(rate=0.5))

imgs = data_layer(name='pixel', size=784)

hidden1 = fc_layer(input=imgs, size=200)
hidden2 = fc_layer(input=hidden1, size=200)

inference = fc_layer(input=hidden2, size=10, act=SoftmaxActivation())

cost = classification_cost(
    input=inference, label=data_layer(
        name='label', size=10))

outputs(cost)
