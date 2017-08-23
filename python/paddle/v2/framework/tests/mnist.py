import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
import numpy
import paddle.v2 as paddle

BATCH_SIZE = 100

scope = core.Scope()
place = core.CPUPlace()
dev_ctx = core.DeviceContext.create(place)

# init_net = core.Net.create()
forward_network = core.Net.create()

# should be init after forward_op is constructed
# backward_net = core.Operator.backward(forward_net, set())
backward_net = None
optimize_net = core.Net.create()


def atom_id():
    id = 0
    while True:
        yield id
        id += 1


uniq_id = atom_id().next


def data_layer(name, dims):
    var = scope.new_var(name)
    tensor = var.get_tensor()
    tensor.set_dims(dims)  # 1 is batch size holder.
    return name


def feed_data(name, data):
    assert isinstance(data, numpy.ndarray)
    tensor = scope.find_var(name).get_tensor()
    tensor.set_dims(data.shape)
    if data.dtype == numpy.dtype('int32'):
        tensor.alloc_int(place)
    elif data.dtype == numpy.dtype('float32'):
        tensor.alloc_float(place)
    else:
        raise ValueError("data type not supported")
    tensor.set(data, place)


def grad_var_name(var_name):
    return var_name + "@GRAD"


def sgd_optimizer(net, param_name, learning_rate=0.01):
    grad_name = grad_var_name(param_name)
    optimize_op = Operator(
        "sgd",
        param=param_name,
        grad=grad_name,
        param_out=param_name,
        learning_rate=learning_rate)
    net.add_op(optimize_op)


# should use operator and add these to the init_network
def init_param(param_name, dims):
    var = scope.new_var(param_name)
    tensor = var.get_tensor()
    tensor.set_dims(dims)
    data = numpy.random.uniform(
        low=0.0, high=1.0, size=tensor.shape()).astype("float32")
    tensor.set(data, place)


# fc_layer
def fc_layer(net, input, size, act="softmax", bias=True, param=None, name=None):
    """
    Add a fc layer to net

    :param input: input variable name.
    :type input: str
    :param size: fully connected layer size.
    :param act: activation name
    :param param: parameter attribute, used for initialize parameters.
    :param bias: bias attribute. False will not have a bias.
    :param name: the name of fc layer. If not set, model will generate a
    readable name
    :return: output variable name.
    """
    if name is None:
        name = 'fc_%d' % uniq_id()
    if not isinstance(name, str):
        raise ValueError("name should be string")

    input_dims = scope.find_var(input).get_tensor().get_dims()

    w_name = param or name + ".w"
    init_param(param_name=w_name, dims=[input_dims[1], size])
    sgd_optimizer(net=optimize_net, param_name=w_name, learning_rate=0.01)

    pre_activation = name + ".mul.out"
    scope.new_var(pre_activation)
    mul_op = Operator("mul", X=input, Y=w_name, Out=pre_activation)
    net.add_op(mul_op)

    # create bias variable if needed
    if bias:
        bias_name = name + ".b"
        init_param(param_name=bias_name, dims=[size])
        sgd_optimizer(
            net=optimize_net, param_name=bias_name, learning_rate=0.01)
        bias_out = name + ".rowwise_add.out"
        scope.new_var(bias_out)
        rowwise_add_op = Operator(
            "rowwise_add", X=pre_activation, b=bias_name, Out=bias_out)
        net.add_op(rowwise_add_op)
        pre_activation = bias_out

    activation_op = Operator(act, X=pre_activation, Y=name)
    net.add_op(activation_op)
    scope.new_var(name)
    net.infer_shape(scope)
    return name


def cross_entropy_layer(net, input, label):
    cost_name = 'cross_entropy_%d' % uniq_id()
    cross_entropy_op = Operator(
        "onehot_cross_entropy", X=input, label=label, Y=cost_name)
    net.add_op(cross_entropy_op)
    scope.new_var(cost_name)
    net.infer_shape(scope)
    return cost_name


def get_backward_net(forward_net):
    net = core.Operator.backward(forward_net, set())
    for input in net.inputs()["all"]:
        var = scope.new_var(input)
        var.get_tensor()
    for output in net.outputs()["all"]:
        var = scope.new_var(output)
        var.get_tensor()
    return net


def print_inputs_outputs(op):
    print("===============" + op.type() + "==============")
    print("***inputs:***")
    for input in op.inputs()["all"]:
        print input, scope.find_var(input).get_tensor().get_dims()
    print("***outputs:***")
    for output in op.outputs()["all"]:
        print output, scope.find_var(output).get_tensor().get_dims()
    print("")
    print("")


def set_cost():
    cost_data = numpy.array(scope.find_var("cross_entropy_1").get_tensor())
    print(cost_data.sum() / len(cost_data))

    cost_grad = scope.find_var(grad_var_name("cross_entropy_1")).get_tensor()

    cost_grad.set_dims(cost_data.shape)
    cost_grad.alloc_float(place)
    cost_grad.set(numpy.ones(cost_data.shape).astype("float32"), place)


images = data_layer(name='pixel', dims=[BATCH_SIZE, 784])
label = data_layer(name='label', dims=[BATCH_SIZE])
fc = fc_layer(net=forward_network, input=images, size=10, act="softmax")
cost = cross_entropy_layer(net=forward_network, input=fc, label=label)

forward_network.complete_add_op(True)
backward_net = get_backward_net(forward_network)
optimize_net.complete_add_op(True)

print(forward_network)
print(backward_net)
print(optimize_net)

print_inputs_outputs(forward_network)
print_inputs_outputs(backward_net)
print_inputs_outputs(optimize_net)

reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE)

PASS_NUM = 1000
for pass_id in range(PASS_NUM):

    print("pass[" + str(pass_id) + "]")
    for data in reader():
        image = numpy.array(map(lambda x: x[0], data)).astype("float32")
        label = numpy.array(map(lambda x: x[1], data)).astype("int32")
        feed_data("pixel", image)
        feed_data("label", label)

        forward_network.infer_shape(scope)
        forward_network.run(scope, dev_ctx)
        set_cost()
        backward_net.infer_shape(scope)
        backward_net.run(scope, dev_ctx)

        optimize_net.run(scope, dev_ctx)
