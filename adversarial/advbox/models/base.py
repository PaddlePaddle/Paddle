"""
The base model of the model.
"""
from abc import ABCMeta
import abc

abstractmethod = abc.abstractmethod


class Model(object):
    """
    Base class of model to provide attack.


    Args:
        bounds(tuple): The lower and upper bound for the image pixel.
        channel_axis(int): The index of the axis that represents the color channel.
        preprocess(tuple): Two element tuple used to preprocess the input. First
            substract the first element, then divide the second element.
    """
    __metaclass__ = ABCMeta

    def __init__(self, bounds, channel_axis, preprocess=None):
        assert len(bounds) == 2
        assert channel_axis in [0, 1, 2, 3]

        if preprocess is None:
            preprocess = (0, 1)
        self._bounds = bounds
        self._channel_axis = channel_axis
        self._preprocess = preprocess

    def bounds(self):
        """
        Return the upper and lower bounds of the model.
        """
        return self._bounds

    def channel_axis(self):
        """
        Return the channel axis of the model.
        """
        return self._channel_axis

    def _process_input(self, input_):
        res = input_
        sub, div = self._preprocess
        if sub != 0:
            res = input_ - sub
        assert div != 0
        if div != 1:
            res /= div
        return res

    @abstractmethod
    def predict(self, image_batch):
        """
        Calculate the prediction of the image batch.

        Args:
            image_batch(numpy.ndarray): image batch of shape (batch_size, height, width, channels).

        Return:
            numpy.ndarray: predictions of the images with shape (batch_size, num_of_classes).
        """
        raise NotImplementedError

    @abstractmethod
    def num_classes(self):
        """
        Determine the number of the classes

        Return:
            int: the number of the classes
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, image_batch):
        """
        Calculate the gradient of the cross-entropy loss w.r.t the image.

        Args:
            image_batch(list): The image and label tuple list.

        Return:
            numpy.ndarray: gradient of the cross-entropy loss w.r.t the image with
                the shape (height, width, channel).
        """
        raise NotImplementedError
