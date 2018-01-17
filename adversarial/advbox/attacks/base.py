#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
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
