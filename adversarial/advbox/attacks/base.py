"""
The base model of the model.
"""
from abc import ABCMeta
#from advbox.base import Model
import abc

abstractmethod = abc.abstractmethod

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

    def __call__(self, image_batch):
        """
        Generate the adversarial sample.

        Args:
        image_batch(list): The image and label tuple list.
        """
        adv_img = self._apply(image_batch)
        return adv_img

    @abstractmethod
    def _apply(self, image_batch):
        """
        Search an adversarial example.

        Args:
        image_batch(list): The image and label tuple list.
        """
        raise NotImplementedError
