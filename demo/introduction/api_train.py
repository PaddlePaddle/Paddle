import paddle.v2 as paddle
import random
import sys


def optimizer_config():
    paddle.config.settings(
        batch_size=12,
        learning_rate=2e-5,
        learning_method=paddle.config.MomentumOptimizer())


def network_config():
    x = paddle.config.data_layer(name='x', size=1)
    y = paddle.config.data_layer(name='y', size=1)
    y_predict = paddle.config.fc_layer(
        input=x,
        param_attr=paddle.config.ParamAttr(name='w'),
        size=1,
        act=paddle.config.LinearActivation(),
        bias_attr=paddle.config.ParamAttr(name='b'))
    cost = paddle.config.regression_cost(input=y_predict, label=y)
    paddle.config.outputs(cost)


def generate_one_batch():
    for i in xrange(2000):
        x = random.random()
        yield {'x': [float(x)], 'y': [2 * x + 0.3]}


def rearrange_input(method, order):
    for item in method():
        retv = []
        for key in order:
            retv.append(item[key])
        yield retv


def main():
    paddle.raw.initPaddle('--use_gpu=false')

    optimizer_proto = paddle.config.parse_optimizer(
        optimizer_conf=optimizer_config)
    optimizer_conf = paddle.raw.OptimizationConfig.createFromProto(
        optimizer_proto)
    __tmp_optimizer__ = paddle.raw.ParameterOptimizer.create(optimizer_conf)
    assert isinstance(__tmp_optimizer__, paddle.raw.ParameterOptimizer)
    enable_types = __tmp_optimizer__.getParameterTypes()

    model_config_proto = paddle.config.parse_network(
        network_conf=network_config)
    gradient_machine = paddle.raw.GradientMachine.createFromConfigProto(
        model_config_proto, paddle.raw.CREATE_MODE_NORMAL, enable_types)
    input_order = model_config_proto.input_layer_names

    updater = paddle.raw.ParameterUpdater.createLocalUpdater(optimizer_conf)
    assert isinstance(updater, paddle.raw.ParameterUpdater)
    assert isinstance(gradient_machine, paddle.raw.GradientMachine)

    gradient_machine.randParameters()

    gradient_machine.start()

    updater.init(gradient_machine)

    updater.startPass()

    out_args = paddle.raw.Arguments.createArguments(0)
    assert isinstance(out_args, paddle.raw.Arguments)

    converter = paddle.data.DataProviderConverter(
        input_types=[paddle.data.dense_vector(1), paddle.data.dense_vector(1)])

    while True:
        data_batch = list(rearrange_input(generate_one_batch, input_order))

        updater.startBatch(len(data_batch))
        in_args = converter(data_batch)

        gradient_machine.forwardBackward(in_args, out_args,
                                         paddle.raw.PASS_TRAIN)

        for param in gradient_machine.getParameters():
            updater.update(param)

        cost_per_instance = out_args.sumCosts() / len(data_batch)

        # print cost_per_instance

        updater.finishBatch(cost=cost_per_instance)

        if cost_per_instance < 1e-5:
            break
        sys.stdout.write('.')

    for param in gradient_machine.getParameters():
        assert isinstance(param, paddle.raw.Parameter)
        v = param.getBuf(paddle.raw.PARAMETER_VALUE)
        assert isinstance(v, paddle.raw.Vector)
        np = v.copyToNumpyArray()
        print param.getName(), np[0]

    updater.finishPass()

    gradient_machine.finish()


if __name__ == '__main__':
    main()
