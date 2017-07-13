import paddle.v2 as paddle
import paddle.v2.dataset.uci_housing as uci_housing
import os

etcd_ip = os.getenv("PADDLE_INIT_MASTER_IP", "127.0.0.1")


def cloud_reader():
    etcd_endpoint = "http://" + etcd_ip + ":2379"
    master_client = paddle.master.client.client(etcd_endpoint, 5, 64)
    master_client.set_dataset("/pfs/public/uci_housing/*")
    for r, e in master_client.next_record():
        if not r:
            break
        yield r


def main():
    # init
    paddle.init(use_gpu=False, trainer_count=1)

    # network config
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
    y_predict = paddle.layer.fc(input=x,
                                param_attr=paddle.attr.Param(name='w'),
                                size=1,
                                act=paddle.activation.Linear(),
                                bias_attr=paddle.attr.Param(name='b'))
    y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
    cost = paddle.layer.mse_cost(input=y_predict, label=y)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(momentum=0)

    #TODO(zhihong) : replace optimizer with new OptimizerConfig

    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer,
                                 is_local=False,
                                 pserver_spec="localhost:2379",
                                 use_etcd=True)

    # event_handler to print training and testing info
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)

        if isinstance(event, paddle.event.EndPass):
            if (event.pass_id + 1) % 10 == 0:
                result = trainer.test(
                    reader=paddle.batch(
                        uci_housing.test(), batch_size=2),
                    feeding={'x': 0,
                             'y': 1})
                print "Test %d, %.2f" % (event.pass_id, result.cost)

    # training
    # NOTE: use uci_housing.train() as reader for non-paddlecloud training
    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(
                cloud_reader, buf_size=500), batch_size=2),
        feeding={'x': 0,
                 'y': 1},
        event_handler=event_handler,
        num_passes=30)


if __name__ == '__main__':
    main()
