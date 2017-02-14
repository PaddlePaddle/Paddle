import paddle.v2.model

__all__ = ['GradientMachine']


class ErrorRateMetric(object):
    pass


class RecallRateMetric(object):
    pass


class GradientMachine(object):
    def __init__(self, model):
        assert isinstance(model, paddle.v2.model.Model)
        self.__share_parameter__(model)
        self.__model__ = model

    def forward(self, data_batch, loss=None):
        """
        Forward a data_batch. If loss is none, this forward not attach with a
        loss function, could be a model inference step. But if loss is not None,
        this forward is a training step, the backward could be invoked, so we
        could get each parameter's gradient.

        :param data_batch:
        :param loss: Loss Configuration. Like CrossEntropy.
        :return: forward result.
        """
        raise NotImplementedError()

    def backward(self, on_parameter_done=None):
        """
        Backward will calculate each parameter's gradient. It should be invoked
        only after forward method. It will return a object, which can get
        each parameter's gradient. The on_parameter_done method will be
        immediately invoked when some parameter has been backward.

        ..  code-block:: python

            gradient_machine.forward(...)

            def on_parameter_done(param, gradient):
                print param.name, "backward done"

            context = gradient_machine.backward(on_parameter_done)

            for each_param in gradient_machine.params():
                each_param -= learning_rate * context.gradient[each_param]

        :param on_parameter_done:
        :return:
        """
        assert on_parameter_done is None or callable(on_parameter_done)
        raise NotImplementedError()

    def test(self, data_batch, metrics=[ErrorRateMetric, RecallRateMetric]):
        """
        test model by some metrics. Like error rate, recall rate, etc.

        This method will return error rate, recall rate as a python object, like
        float value.

        ..  code-block:: python

            error_rate, recall_rate = gradient_machine.test(data_batch,
                        metrics=[ErrorRateMetric, RecallRateMetric])

            print 'Error rate %f, Recall %f' %(error_rate, recall_rate)


        :param data_batch:
        :param metrics:
        :return:
        """
        raise NotImplementedError()

    def get_parameters(self):
        """
        Get all parameters
        :return:
        """
        return self.__model__.parameters()

    def __share_parameter__(self, model):
        """
        A GradientMachine will share parameters from model.
        :return:
        """
        raise NotImplementedError()
