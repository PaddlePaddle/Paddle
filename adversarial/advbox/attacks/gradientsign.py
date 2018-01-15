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
This module provide the attack method for FGSM's implement.
"""
from __future__ import division
import numpy as np
from collections import Iterable
from .base import Attack


class GradientSignAttack(Attack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm (and is known as the "Fast Gradient Sign Method"). This is therefore called
    the Fast Gradient Method.
    Paper link: https://arxiv.org/abs/1412.6572
    """

    def _apply(self, image_label, epsilons=1000):
        assert len(image_label) == 1
        pre_label = np.argmax(self.model.predict(image_label))

        min_, max_ = self.model.bounds()
        gradient = self.model.gradient(image_label)
        gradient_sign = np.sign(gradient) * (max_ - min_)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)

        for epsilon in epsilons:
            adv_img = image_label[0][0].reshape(
                gradient_sign.shape) + epsilon * gradient_sign
            adv_img = np.clip(adv_img, min_, max_)
            adv_label = np.argmax(self.model.predict([(adv_img, 0)]))
            if pre_label != adv_label:
                return adv_img


FGSM = GradientSignAttack
