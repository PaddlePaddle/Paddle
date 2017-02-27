import py_paddle.swig_paddle as api

__all__ = ['ITester', 'PlainLocalTrainingTester']


class ITester(object):
    """
    Tester interface.

    Tester is using for model testing. It uses topology, parameters and a data
    reader creator as parameters to construct. In `test` method, the user can
    control test batch_size and test batches. The test function will return
    metrics defined in model topology.
    """

    def __init__(self, topology, parameters, reader_creator):
        self.__topology__ = topology
        self.__parameters__ = parameters
        self.__reader_creator__ = reader_creator

    def test(self, batch_size, num_batches=0):
        """
        test will test neural network by `num_batches`, default is test whole
        testing data.

        :param batch_size: Batch size for this testing method.
        :param num_batches: number of batches for testing. <=0 means test whole
                            data.
        :return: Test Result.
        """
        raise NotImplementedError


class PlainLocalTrainingTester(ITester):
    """
    :type __gradient_machine__: api.GradientMachine
    """

    def __init__(self, gradient_machine, topology, parameters, reader_creator):
        super(PlainLocalTrainingTester, self).__init__(topology, parameters,
                                                       reader_creator)
        self.__gradient_machine__ = gradient_machine

    def test(self, batch_size, num_passes=1):
        raise NotImplementedError("not implemented now, but will implemented "
                                  "when mnist data complete.")
