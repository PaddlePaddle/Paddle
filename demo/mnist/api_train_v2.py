from paddle.trainer_config_helpers import *
from paddle.trainer.PyDataProvider2 import dense_vector, integer_value
import paddle.v2 as paddle
import numpy
import mnist_util


def train_reader():
    train_file = './data/raw_data/train'
    generator = mnist_util.read_from_mnist(train_file)
    for item in generator:
        yield item


def network_config():
    imgs = data_layer(name='pixel', size=784)
    hidden1 = fc_layer(input=imgs, size=200)
    hidden2 = fc_layer(input=hidden1, size=200)
    inference = fc_layer(input=hidden2, size=10, act=SoftmaxActivation())
    cost = classification_cost(
        input=inference, label=data_layer(
            name='label', size=10))
    outputs(cost)


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    model_config = parse_network_config(network_config)
    parameters = paddle.parameters.create(model_config)
    for param_name in parameters.keys():
        array = parameters[param_name]
        array[:] = numpy.random.uniform(low=-1.0, high=1.0, size=array.shape)
        parameters[param_name] = array

    adam_optimizer = paddle.optimizer.Optimizer(
        learning_rate=0.01, learning_method=AdamOptimizer())

    def event_handler(event):
        if isinstance(event, paddle.trainer.EndIteration):
            para = parameters['___fc_layer_2__.w0']
            print "Pass %d, Batch %d, Cost %f, Weight Mean Of Fc 2 is %f" % (
                event.pass_id, event.batch_id, event.cost, para.mean())

        else:
            pass

    trainer = paddle.trainer.SGDTrainer(update_equation=adam_optimizer)

    trainer.train(train_data_reader=train_reader,
                  topology=model_config,
                  parameters=parameters,
                  event_handler=event_handler,
                  batch_size=32,  # batch size should be refactor in Data reader
                  data_types={  # data_types will be removed, It should be in
                      # network topology
                      'pixel': dense_vector(784),
                      'label': integer_value(10)
                  })


if __name__ == '__main__':
    main()
