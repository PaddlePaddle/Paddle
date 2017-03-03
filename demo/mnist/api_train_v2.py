import paddle.v2 as paddle


def main():
    paddle.init(use_gpu=False, trainer_count=1)

    # define network topology
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))
    hidden1 = paddle.layer.fc(input=images, size=200)
    hidden2 = paddle.layer.fc(input=hidden1, size=200)
    inference = paddle.layer.fc(input=hidden2,
                                size=10,
                                act=paddle.activation.Softmax())
    cost = paddle.layer.classification_cost(input=inference, label=label)

    parameters = paddle.parameters.create(cost)

    adam_optimizer = paddle.optimizer.Adam(learning_rate=0.01)

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=adam_optimizer)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1000 == 0:
                result = trainer.test(paddle.dataset.mnist.test(), 256)

                print "Pass %d, Batch %d, Cost %.2f, %s\n" \
                      "Testing cost %.2f metrics %s" % (
                          event.pass_id, event.batch_id, event.cost,
                          event.metrics,
                          result.cost, result.metrics)
        else:
            pass

    trainer.train(
        reader=paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=8192),
        batch_size=32,
        event_handler=event_handler)

    # output is a softmax layer. It returns probabilities.
    # Shape should be (100, 10)
    probs = paddle.infer(
        output=inference,
        parameters=parameters,
        reader=paddle.reader.firstn(
            paddle.dataset.mnist.test(), n=100),
        reader_dict={'pixel': 0},
        batch_size=32)
    print probs.shape


if __name__ == '__main__':
    main()
