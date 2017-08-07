import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
import numpy


class ParameterAttribute(object):
    def __init__(self, name=None, initial_min=-1.0, initial_max=1.0):
        self.name = name
        self.initial_min = initial_min
        self.initial_max = initial_max


class Model(object):
    def __init__(self, place=core.CPUPlace()):
        self.scope = core.Scope()
        # init network is the network with initialize operators, they are random
        # operators or zero operators.
        self.init_network = core.Net.create()

        # forward network is the full network of all defined forward operators.
        # We can run a subnet from it, by `get_subnet` method.
        self.forward_network = core.Net.create()

        # A counter to give a readable variable name in one model
        self.name_counter = 0

        # A place that this model run on.
        self.place = place

        # The device context of that place.
        self.dev_ctx = core.DeviceContext.create(place)

    def data_layer(self, name, dims):
        """
        Add a data layer for this model.
        
        :param name: the data layer name, i.e., the output variable name. 
        :param dims: the dimension without batch size.
        :return: Output variable name.
        :rtype: str
        """
        if isinstance(dims, int):
            dims = [dims]

        if not isinstance(dims, list) or isinstance(dims, tuple):
            raise ValueError("dims should be an int, list or tuple.")

        self.assert_not_create_var(name)

        var = self.scope.new_var(name)
        tensor = var.get_tensor()
        tensor.set_dims([1] + dims)  # 1 is batch size holder.

        return name

    def fc_layer(self,
                 input,
                 size,
                 act="sigmoid",
                 param=None,
                 bias=True,
                 name=None):
        """
        Add a fc layer to model
        
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
            name = 'fc_%d' % self.name_counter
            self.name_counter += 1
        if not isinstance(name, str):
            raise ValueError("name should be string")

        self.assert_not_create_var(name)
        input_dims = self.scope.find_var(input).get_tensor().get_dims()

        param_name = self.add_param_to_init_network(
            default_name=name + ".w", param=param, dims=[input_dims[1], size])

        pre_activation = name + ".mul.out"
        self.scope.new_var(pre_activation)
        mul_op = Operator("mul", X=input, Y=param_name, Out=pre_activation)
        self.forward_network.add_op(mul_op)

        if bias:
            bias_name = self.add_param_to_init_network(
                default_name=name + ".b", param=bias, dims=[size])
            bias_out = name + ".rowwise_add.out"
            self.scope.new_var(bias_out)
            rowwise_add_op = Operator(
                "rowwise_add", X=pre_activation, b=bias_name, Out=bias_out)
            self.forward_network.add_op(rowwise_add_op)
            pre_activation = bias_out

        out = Operator(act, X=pre_activation, Y=name)
        self.forward_network.add_op(out)
        self.scope.new_var(name)
        # TODO(yuyang): InferShape only these operator.
        self.forward_network.infer_shape(self.scope)
        return name

    def init_params(self):
        """
        Initialize all parameters inside this model
        """
        self.init_network.infer_shape(self.scope)
        self.init_network.run(self.scope, self.dev_ctx)

    def forward(self, data, output=None):
        """
        Forward a (sub) network
        :param data: the data layer's data. Should be a dict, 
        key is variable name, value is a numpy array.
        :param output: The output variable name. If not set, full forward 
        network will be run. If set, only operators could generate that output
         will be run, i.e., only a sub network will be run.
        :return: Nothing, use scope to get variable.
        """
        for k in data:
            arr = data[k]
            if not isinstance(arr, numpy.ndarray):
                raise TypeError(
                    "data should be a dict, value should be ndarray")

            var = self.scope.find_var(k)
            tensor = var.get_tensor()
            tensor.set_dims(arr.shape)
            tensor.alloc_float(self.place)
            tensor.set(arr, self.place)

        net = self.get_subnet(self.forward_network, output)
        net.infer_shape(self.scope)
        net.run(self.scope, self.dev_ctx)

    def get_subnet(self, full_net, output):
        """
        Get a sub network from a full_network which is necessary to generate 
        output variable
        """
        if output is None:
            return full_net
        else:
            if not isinstance(output, str):
                raise TypeError("output of get_subnet should be str")

            all_operators = full_net.ops()
            need_var_set = set()
            need_var_set.add(output)

            ret_ops = []

            for i in xrange(len(all_operators)):
                op = all_operators[-i]
                outs = op.outputs()
                if Model.any_in_set(outs, need_var_set):
                    for out in outs:
                        need_var_set.add(out)
                    for ipt in op.inputs():
                        need_var_set.add(ipt)

                    ret_ops.append(op)

            if len(ret_ops) == 0:
                raise TypeError("Cannot calculate variable %s in this network",
                                output)

            net = core.Net.create()
            for i in xrange(len(ret_ops)):
                net.add_op(ret_ops[-i])
            return net

    @staticmethod
    def any_in_set(names, name_set):
        """
        Returns true if any element in set.
        """
        for name in names:
            if name in name_set:
                return True
        return False

    def add_param_to_init_network(self, default_name, param, dims):
        """
        Add parameter to initialize network
        """
        if not isinstance(param, ParameterAttribute):
            param = ParameterAttribute()

        if param.name is not None:
            name = param.name
        else:
            name = default_name
        var = self.scope.new_var(name)
        tensor = var.get_tensor()
        tensor.set_dims(dims)
        init_op = Operator(
            "uniform_random",
            Out=name,
            dims=dims,
            min=param.initial_min,
            max=param.initial_max)
        self.init_network.add_op(init_op)
        return name

    def assert_not_create_var(self, name):
        """
        Assert a variable has not been created.
        """
        var = self.scope.find_var(name)
        if var is not None:
            raise ValueError("Variable %s has been created in this model", name)


g_model = Model()


# The function below can be automatically generated in runtime.
def data_layer(*args, **kwargs):
    return g_model.data_layer(*args, **kwargs)


def fc_layer(*args, **kwargs):
    return g_model.fc_layer(*args, **kwargs)


def init_params():
    g_model.init_params()


def forward(*args, **kwargs):
    g_model.forward(*args, **kwargs)
