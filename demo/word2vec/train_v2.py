import math

import paddle.v2 as paddle

dictsize = 1953
embsize = 32
hiddensize = 256
N = 5


def wordemb(inlayer):
    wordemb = paddle.layer.table_projection(
        input=inlayer,
        size=embsize,
        param_attr=paddle.attr.Param(
            name="_proj",
            initial_std=0.001,
            learning_rate=1,
            l2_rate=0, ))
    return wordemb


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)
    firstword = paddle.layer.data(
        name="firstw", type=paddle.data_type.integer_value(dict_size))
    secondword = paddle.layer.data(
        name="secondw", type=paddle.data_type.integer_value(dict_size))
    thirdword = paddle.layer.data(
        name="thirdw", type=paddle.data_type.integer_value(dict_size))
    fourthword = paddle.layer.data(
        name="fourthw", type=paddle.data_type.integer_value(dict_size))
    nextword = paddle.layer.data(
        name="fifthw", type=paddle.data_type.integer_value(dict_size))

    Efirst = wordemb(firstword)
    Esecond = wordemb(secondword)
    Ethird = wordemb(thirdword)
    Efourth = wordemb(fourthword)

    contextemb = paddle.layer.concat(input=[Efirst, Esecond, Ethird, Efourth])
    hidden1 = paddle.layer.fc(input=contextemb,
                              size=hiddensize,
                              act=paddle.activation.Sigmoid(),
                              layer_attr=paddle.attr.Extra(drop_rate=0.5),
                              bias_attr=paddle.attr.Param(learning_rate=2),
                              param_attr=paddle.attr.Param(
                                  initial_std=1. / math.sqrt(embsize * 8),
                                  learning_rate=1))
    predictword = paddle.layer.fc(input=hidden1,
                                  size=dict_size,
                                  bias_attr=paddle.attr.Param(learning_rate=2),
                                  act=paddle.activation.Softmax())

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                result = trainer.test(
                    paddle.batch(
                        paddle.dataset.imikolov.test(word_dict, N), 32))
                print "Pass %d, Batch %d, Cost %f, %s, Testing metrics %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics,
                    result.metrics)

    cost = paddle.layer.classification_cost(input=predictword, label=nextword)
    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=3e-3,
        regularization=paddle.optimizer.L2Regularization(8e-4))
    trainer = paddle.trainer.SGD(cost, parameters, adam_optimizer)
    trainer.train(
        paddle.batch(paddle.dataset.imikolov.train(word_dict, N), 32),
        num_passes=30,
        event_handler=event_handler)


if __name__ == '__main__':
    main()
