import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
import numpy
import paddle.v2 as paddle
exit(
    0
)  # FIXME(yuyang18): InferShape has been removed, this unittest should be changed until compile time is ready

BATCH_SIZE = 100

scope = core.Scope()
place = core.CPUPlace()
# if you want to test GPU training, you can use gpu place
# place = core.GPUPlace(0)
dev_ctx = core.DeviceContext.create(place)

init_net = core.Net.create()
forward_net = core.Net.create()
backward_net = None
optimize_net = core.Net.create()


def atomic_id():
    id = 0
    while True:
        yield id
        id += 1


uniq_id = atomic_id().next


def data_layer(name, dims):
    var = scope.new_var(name)
    tensor = var.get_tensor()
    tensor.set_dims(dims)  # 1 is batch size holder.
    return name


def feed_data(name, data):
    assert isinstance(data, numpy.ndarray)
    tensor = scope.find_var(name).get_tensor()
    tensor.set_dims(data.shape)
    if data.dtype == numpy.dtype("int32"):
        tensor.alloc_int(place)
    elif data.dtype == numpy.dtype("float32"):
        tensor.alloc_float(place)
    else:
        raise ValueError("data type not supported")
    tensor.set(data, place)


def grad_var_name(var_name):
    return var_name + "@GRAD"


def sgd_optimizer(net, param_name, learning_rate=0.005):
    grad_name = grad_var_name(param_name)
    optimize_op = Operator(
        "sgd",
        param=param_name,
        grad=grad_name,
        param_out=param_name,
        learning_rate=learning_rate)
    net.append_op(optimize_op)


# should use operator and add these to the init_network
def init_param(net, param_name, dims):
    scope.new_var(param_name)
    op = Operator(
        "uniform_random", Out=param_name, dims=dims, min=-0.5, max=0.5, seed=10)
    op.infer_shape(scope)
    net.append_op(op)


# fc_layer
def fc_layer(net, input, size, act="softmax", bias=True, param=None, name=None):
    """
    The fully connected layer.

    :param input: The name of input variable.
    :type input: str
    :param size: The size of fully connected layer.
    :param act: The name of activation.
    :param param: The attribute of learnable parameter which can be used to
                  modify initialization mean and std of the parameter.
    :param bias: The attribute of bias. If set False, this layer does not have
                 a bias.
    :param name: The name of this layer. If it is not set explictly, a name
                 will be generated automatically.
    :return: The name of the output variable.
    """

    if name is None:
        name = "fc_%d" % uniq_id()
    if not isinstance(name, str):
        raise ValueError("The name of a layer should be a string.")

    input_dims = scope.find_var(input).get_tensor().get_dims()

    w_name = param or name + ".w"
    init_param(net=init_net, param_name=w_name, dims=[input_dims[1], size])
    sgd_optimizer(net=optimize_net, param_name=w_name, learning_rate=0.01)

    pre_activation = name + ".mul.out"
    scope.new_var(pre_activation)
    mul_op = Operator("mul", X=input, Y=w_name, Out=pre_activation)
    net.append_op(mul_op)

    # create bias variable if needed
    if bias:
        bias_name = name + ".b"
        init_param(net=init_net, param_name=bias_name, dims=[size])
        sgd_optimizer(
            net=optimize_net, param_name=bias_name, learning_rate=0.001)
        bias_out = name + ".rowwise_add.out"
        scope.new_var(bias_out)
        rowwise_append_op = Operator(
            "rowwise_add", X=pre_activation, b=bias_name, Out=bias_out)
        net.append_op(rowwise_append_op)
        pre_activation = bias_out

    activation_op = Operator(act, X=pre_activation, Y=name)
    net.append_op(activation_op)
    scope.new_var(name)
    net.infer_shape(scope)
    return name


def cross_entropy_layer(net, input, label):
    cost_name = "cross_entropy_%d" % uniq_id()
    cross_entropy_op = Operator(
        "cross_entropy", X=input, Label=label, Y=cost_name)
    net.append_op(cross_entropy_op)
    scope.new_var(cost_name)
    net.infer_shape(scope)
    return cost_name


def create_backward_net(forward_net):
    net = core.Operator.backward(forward_net, set())
    for input in net.inputs()["all"]:
        var = scope.new_var(input)
        var.get_tensor()
    for output in net.outputs()["all"]:
        var = scope.new_var(output)
        var.get_tensor()
    return net


def debug_print_op(op):
    print("===============" + op.type() + "==============")
    print("***inputs:***")
    for input in op.inputs()["all"]:
        print input, scope.find_var(input).get_tensor().get_dims()
    print("\n***outputs:***")
    for output in op.outputs()["all"]:
        print output, scope.find_var(output).get_tensor().get_dims()
    print("")
    print("")


def set_cost(cost):
    cost_shape = numpy.array(scope.find_var(cost).get_tensor()).shape
    cost_grad = \
        scope.find_var(grad_var_name(cost)).get_tensor()
    cost_grad.set_dims(cost_shape)
    cost_grad.alloc_float(place)
    cost_grad.set(numpy.ones(cost_shape).astype("float32"), place)


def get_cost_mean(cost):
    cost_data = numpy.array(scope.find_var(cost).get_tensor())
    return cost_data.sum() / len(cost_data)


def error_rate(predict, label):
    predict_var = numpy.array(scope.find_var(predict).get_tensor()).argmax(
        axis=1)
    label = numpy.array(scope.find_var(label).get_tensor())
    error_num = numpy.sum(predict_var != label)
    return error_num / float(len(label))


images = data_layer(name="pixel", dims=[BATCH_SIZE, 784])
labels = data_layer(name="label", dims=[BATCH_SIZE, 1])
fc1 = fc_layer(net=forward_net, input=images, size=100, act="sigmoid")
fc2 = fc_layer(net=forward_net, input=fc1, size=100, act="sigmoid")
predict = fc_layer(net=forward_net, input=fc2, size=10, act="softmax")
cost = cross_entropy_layer(net=forward_net, input=predict, label=labels)

init_net.complete_add_op(True)
forward_net.complete_add_op(True)
backward_net = create_backward_net(forward_net)
optimize_net.complete_add_op(True)

print(init_net)
print(forward_net)
print(backward_net)
print(optimize_net)

debug_print_op(forward_net)
debug_print_op(backward_net)
debug_print_op(optimize_net)

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE)


def test(cost_name):
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
    cost = []
    error = []
    for data in test_reader():
        image_data = numpy.array(map(lambda x: x[0], data)).astype("float32")
        label_data = numpy.array(map(lambda x: x[1], data)).astype("int32")
        label_data = numpy.expand_dims(label_data, axis=1)
        feed_data(images, image_data)
        feed_data(labels, label_data)

        forward_net.infer_shape(scope)
        forward_net.run(scope, dev_ctx)
        cost.append(get_cost_mean(cost_name))
        error.append(error_rate(predict, "label"))
    print("cost=" + str(sum(cost) / float(len(cost))) + " error_rate=" + str(
        sum(error) / float(len(error))))


PASS_NUM = 1

init_net.run(scope, dev_ctx)
for pass_id in range(PASS_NUM):
    batch_id = 0

    for data in train_reader():
        image_data = numpy.array(map(lambda x: x[0], data)).astype("float32")
        label_data = numpy.array(map(lambda x: x[1], data)).astype("int32")
        label_data = numpy.expand_dims(label_data, axis=1)
        feed_data(images, image_data)
        feed_data(labels, label_data)

        forward_net.infer_shape(scope)
        forward_net.run(scope, dev_ctx)
        set_cost(cost)
        backward_net.infer_shape(scope)
        backward_net.run(scope, dev_ctx)

        optimize_net.run(scope, dev_ctx)
        if batch_id % 100 == 0:
            print("pass[" + str(pass_id) + "] batch_id[" + str(batch_id) + "]")
            test(cost)

        batch_id = batch_id + 1
