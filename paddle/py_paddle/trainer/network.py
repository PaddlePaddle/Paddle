from paddle.trainer_config_helpers import *
from paddle.trainer_config_helpers import inputs as ipts
import paddle.trainer.PyDataProvider2 as dp2

__all__ = ['NetworkConfig', 'network']


class NetworkConfig(object):
    """
    Base class for a neural network configuration.

    NOTE: this object only hold neural network's compute graph, not hold any
    parameters.
    """

    def __init__(self):
        pass

    def input_order(self):
        """
        Input Order is the order of neural network's data layer.

        The gradient_machine's input arguments list should be the same order of
        input order.
        :return: list of data layer name.
        :rtype: list
        """
        raise NotImplemented()

    def input_types(self):
        """
        Input types of each data layer.
        :return: a dict. Key is data layer's name. Value is the type of this
                         data layer.
        :rtype: dict
        """
        raise NotImplemented()

    def network_graph(self):
        """
        get the ModelConfig of this neural network. Return raw protobuf object.
        :return: ModelConfig protobuf object.
        :rtype: paddle.proto.ModelConfig_pb2.ModelConfig
        """
        raise NotImplemented()

    def optimize_graph(self):
        """
        get the OptimizationConfig of this neural network. Return raw protobuf
        object.
        :return: OptimizationConfig protobuf object.
        :rtype: paddle.proto.TrainerConfig_pb2.OptimizationConfig
        """
        raise NotImplemented()

    def provider(self, **kwargs):
        return dp2.provider(input_types=self.input_types(), **kwargs)


def network(inputs, **opt_kwargs):
    """
    A decorator for neural network method. It will wrap a method to a
    NetworkConfig.

    ..  code-block: python

        @network(inputs={'img': dense_vector(784), 'label':integer_value(10)},
            batch_size=1000, learning_rate=1e-3, learning_method=AdamOptimizer())
        def mnist_network(img, label):
            hidden = fc_layer(input=img, size=200)
            ...
            cost = classification_cost(input=inference, label=label)
            return cost

        mnist = mnist_network()

    :param inputs: input dictionary for this neural network. The key of this
    dictionary is wrapped method parameter. Value is data type.
    :param opt_kwargs: Other arguments of this method are passed to optimizers.
    :return: NetworkConfig Class.
    :rtype: class
    """

    def __impl__(func):
        class NetworkConfigImpl(NetworkConfig):
            def __init__(self):
                NetworkConfig.__init__(self)
                self.__inputs__ = inputs
                self.__network_graph__ = None
                self.__optimize_graph__ = None

            def input_order(self):
                return inputs.keys()

            def input_types(self):
                return self.__inputs__

            def network_graph(self):
                if self.__network_graph__ is None:

                    def __network_graph_func__():
                        kwargs = dict()
                        lst = list()
                        for k in inputs:
                            v = inputs[k]
                            data = data_layer(name=k, size=v.dim)
                            kwargs[k] = data
                            lst.append(data)
                        ipts(*lst)
                        rst = func(**kwargs)
                        if not isinstance(rst, tuple):
                            rst = (rst, )
                        outputs(*rst)

                    self.__network_graph__ = parse_network_config(
                        __network_graph_func__)
                return self.__network_graph__

            def optimize_graph(self):
                if self.__optimize_graph__ is None:

                    def __optimize_graph_func__():
                        settings(**opt_kwargs)

                    self.__optimize_graph__ = parse_optimizer_config(
                        __optimize_graph_func__)
                return self.__optimize_graph__

        return NetworkConfigImpl

    return __impl__
