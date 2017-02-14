from paddle.v2.model import Model

__all__ = ['Evaluator']


class Evaluator(object):
    def __init__(self, model):
        if not isinstance(model, Model):
            raise ValueError("model must be python.model.Model type")

        self.model = model

    def forward(self, data_batch, cost=None, metrics=None, begin=None,
                end=None):
        """
        forward neural network and get forwarding result.

        ..  code-block:: python

            result = evaluator.forward(input_data,
                                       cost=CrossEntropy(input_layer='predict',
                                                         label=label_data),
                                       metrics=[ErrorRate(input_layer='predict',
                                                         label=label_data),
                                                RecallRate(input_layer='predict',
                                                         label=label_data)])

            print result['cost']
            print result['metrics'][1]
            print evaluator.activation("some_layer_name")

        :param data_batch: The input data in mini-batch
        :param cost: The cost function of Neural Network. It could be a Paddle
                     cost layer type.
        :param metrics: The metric function of Neural Network.
        :param begin: The start layer of neural network forward. Default is None
                      which means begin with data_layer
        :param end: The end layer of neural network forward. Default is None.
                    which means forward all layers.
        :return: result dictionary, which keys are 'cost', 'metrics'
        """
        raise NotImplementedError()
        # Stub return value
        return {'cost': 27.32, 'metrics': [0.93, 0.21]}

    def activation(self, layer):
        """
        Get activation of the layer
        """
        raise NotImplementedError()
