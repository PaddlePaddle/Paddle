import py_paddle.swig_paddle as api
from paddle.proto.ModelConfig_pb2 import ModelConfig

__all__ = ['Model', 'create']


class Model(object):
    """
    Model stores neural network topology(which is protobuf) and parameters. It
    could be serialized and deserialized.

    Details usage see the methods comments.
    """

    def __init__(self, model_config):
        """
        Constructor.

        :note: Use paddle.model.create instead. This method should not be
               invoked out of this package.
        :param model_config: neural network topology (in protobuf)
        """
        assert isinstance(model_config, ModelConfig)
        self.__model_config__ = model_config
        self.__neural_network__ = api.GradientMachine.createByModelConfig(
            model_config)
        assert isinstance(self.__neural_network__, api.GradientMachine)

    def get_topology(self):
        """
        Get neural network topology in protobuf.

        Example.

        ..  code-block:: python

            topology = model.get_topology()
            for layer in topology.layers:
                print layer.name

        :rtype: ModelConfig
        """
        return self.__model_config__

    def get_parameter(self, param_name):
        """
        Get Parameter which could be read or write.

        Example.

        ..  code-block:: python

            param = model.get_parameter('embedding')
            embedding = param[34:]
            print 'Word id %d, embedding %s'%(34, embedding)
            param[34:] = 0.0  # zero this embedding.

        :param param_name:
        :return: Paddle Status
        """

        assert isinstance(param_name, basestring)
        raise NotImplementedError()

    def randomize_parameter(self):
        """
        Randomize parameter by network topology.
        :return: Paddle Status
        """
        raise NotImplementedError()

    def serialize(self, fp):
        """
        Serialize whole model to stream.

        ..  code-block:: python

            ok = model.serialize(tcp_stream)
            assert ok

        :param fp: a file like object. In python, file like object could be a
                   general stream.
        :type fp: file like
        :return: Paddle Status
        """

        assert hasattr(fp, 'write'), "fp should be file like object(stream)."
        raise NotImplementedError()

    @staticmethod
    def deserialize(fp):
        """
        Deserialize from a stream and create a new model.

        ..  code-block:: python

            model = paddle.model.Model.deserialize(tcp_stream)

        :param fp: a file like object.
        :type fp: file like.
        :return:
        :rtype: Model
        """
        assert hasattr(fp, 'read'), 'fp should be file like object(stream).'
        raise NotImplementedError()


def create(*args):
    """
    Create Model by network topology, for example:

    ..  code-block:: python

        import paddle.v2 as paddle

        data = paddle.layers.data(...)
        ...
        prediction = paddle.layers.fc(...)

        model = paddle.model.create(prediction)


    :param args: layer outputs. a list of layer output
    :type args: list of layers
    :rtype: Model
    """
    raise NotImplementedError()
