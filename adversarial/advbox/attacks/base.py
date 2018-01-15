"""
The base model of the model.
"""
from abc import ABCMeta, abstractmethod


class Attack(object):
    """
    Abstract base class for adversarial attacks. `Attack` represent an adversarial attack
    which search an adversarial example. subclass should implement the _apply() method.

    Args:
        model(Model): an instance of the class advbox.base.Model.

    """
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    def __call__(self, image_label):
        """
        Generate the adversarial sample.

        Args:
        image_label(list): The image and label tuple list with one element.
        """
        adv_img = self._apply(image_label)
        return adv_img

    @abstractmethod
    def _apply(self, image_label):
        """
        Search an adversarial example.

        Args:
        image_batch(list): The image and label tuple list with one element.
        """
        raise NotImplementedError
