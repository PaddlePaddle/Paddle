from paddle.proto.ModelConfig_pb2 import ModelConfig

__all__ = ["Model"]


class Model(object):
    def __init__(self, network_topology):
        """
        Constructor.
        :param network_topology:
        """
        if not isinstance(network_topology, ModelConfig):
            raise ValueError(
                "network_topology should be ModelConfig protobuf object")
        self.__topology__ = network_topology

    def serialize(self, fp):
        """
        Serialize to stream
        :param fp:
        :return:
        """
        if not hasattr(fp, 'write'):
            raise ValueError("fp must be writable")
        raise NotImplementedError()

    @staticmethod
    def deserialize(fp):
        if not hasattr(fp, 'read'):
            raise ValueError("fp must be readable")
        raise NotImplementedError()

    def get_parameter(self, parameter_name):
        if not isinstance(parameter_name, basestring):
            raise ValueError("parameter_name should be string")
        raise NotImplementedError()

    def get_topology(self):
        raise NotImplementedError()
