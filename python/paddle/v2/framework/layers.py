from paddle.v2.framework.model import *
import numpy


def is_list_like(o):
    return isinstance(o, list) or isinstance(o, tuple)


def data(name, shape, model=None):
    if model is None:
        model = g_model

    if isinstance(shape, int):
        shape = [shape]

    if not is_list_like(shape):
        raise ValueError()

    if id(model.cur_scope) != id(model.global_scope):
        raise ValueError("Data Layer must be declared in global scope")

    tensor = model.cur_scope.new_var(name).get_tensor()
    tensor.set_dims([1] + shape)
    return name


def fc(input,
       size,
       name=None,
       param_attr=None,
       bias_attr=False,
       act="sigmoid",
       model=None):
    if model is None:
        model = g_model

    if name is None:
        name = model.next_name('fc')

    if param_attr is None:
        param_attr = ParameterAttribute.default_weight_attr()

    dim = model.cur_scope.find_var(input).get_tensor().get_dims()
    w = model.create_parameter(name + ".weight.", param_attr, [dim[1], size])

    tmp = model.add_op_and_infer_shape("mul", X=input, Y=w)

    if bias_attr is None or bias_attr is True:
        bias_attr = ParameterAttribute.default_bias_attr()
    if bias_attr:
        b = model.create_parameter(name + ".bias.", bias_attr, [size])
        tmp = model.add_op_and_infer_shape('rowwise_add', X=tmp, b=b)

    if act:
        tmp = model.add_op_and_infer_shape(act, X=tmp)

    return tmp


if __name__ == '__main__':
    x = data("X", shape=784)
    hidden = fc(x, size=100, bias_attr=True)
    hidden = fc(hidden, size=100, bias_attr=True)
    prediction = fc(hidden, size=10, bias_attr=True, act='softmax')

    g_model.init_parameters()
    g_model.feed_data({"X": numpy.random.random((1000, 784)).astype('float32')})

    for i in xrange(1000):
        g_model.run()
        print numpy.array(g_model.find_tensor(prediction)).mean()
