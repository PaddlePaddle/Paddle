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

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
        else:
            pass

    trainer = paddle.trainer.SGD(update_equation=adam_optimizer)

    trainer.train(
        reader=paddle.reader.batched(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train_creator(), buf_size=8192),
            batch_size=32),
        cost=cost,
        parameters=parameters,
        event_handler=event_handler,
        batch_size=32,  # batch size should be refactor in Data reader
        reader_dict={images.name: 0,
                     label.name: 1})


if __name__ == '__main__':
    main()
