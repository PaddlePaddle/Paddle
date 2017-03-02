import paddle.v2 as paddle
import cPickle


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

    try:
        with open('params.pkl', 'r') as f:
            parameters = cPickle.load(f)
    except IOError:
        parameters = paddle.parameters.create(cost)

    adam_optimizer = paddle.optimizer.Adam(learning_rate=0.01)

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=adam_optimizer)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1000 == 0:
                result = trainer.test(reader=paddle.reader.batched(
                    paddle.dataset.mnist.test(), batch_size=256))

                print "Pass %d, Batch %d, Cost %f, %s, Testing metrics %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics,
                    result.metrics)

                with open('params.pkl', 'w') as f:
                    cPickle.dump(
                        parameters, f, protocol=cPickle.HIGHEST_PROTOCOL)

        else:
            pass

    trainer.train(
        reader=paddle.reader.batched(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=32),
        event_handler=event_handler)


if __name__ == '__main__':
    main()
