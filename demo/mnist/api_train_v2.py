import numpy
import paddle.v2 as paddle
from paddle.trainer.PyDataProvider2 import dense_vector, integer_value

import mnist_util


def train_reader():
    train_file = './data/raw_data/train'
    generator = mnist_util.read_from_mnist(train_file)
    for item in generator:
        yield item


def main():
    paddle.init(use_gpu=False, trainer_count=1)

    # define network topology
    images = paddle.layer.data(name='pixel', size=784)
    label = paddle.layer.data(name='label', size=10)
    hidden1 = paddle.layer.fc(input=images, size=200)
    hidden2 = paddle.layer.fc(input=hidden1, size=200)
    inference = paddle.layer.fc(input=hidden2,
                                size=10,
                                act=paddle.activation.Softmax())
    cost = paddle.layer.classification_cost(input=inference, label=label)

    topology = paddle.layer.parse_network(cost)
    parameters = paddle.parameters.create(topology)
    for param_name in parameters.keys():
        array = parameters.get(param_name)
        array[:] = numpy.random.uniform(low=-1.0, high=1.0, size=array.shape)
        parameters.set(parameter_name=param_name, value=array)

    adam_optimizer = paddle.optimizer.Adam(learning_rate=0.01)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            para = parameters.get('___fc_2__.w0')
            print "Pass %d, Batch %d, Cost %f, Weight Mean Of Fc 2 is %f" % (
                event.pass_id, event.batch_id, event.cost, para.mean())

        else:
            pass

    trainer = paddle.trainer.SGD(update_equation=adam_optimizer)

    trainer.train(train_data_reader=train_reader,
                  topology=topology,
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
