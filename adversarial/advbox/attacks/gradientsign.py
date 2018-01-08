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

    def _apply(self, image_batch, epsilons=1000):
        pre_label = np.argmax(self.model.predict(image_batch))

        min_, max_ = self.model.bounds()
        gradient = self.model.gradient(image_batch)
        gradient_sign = np.sign(gradient) * (max_ - min_)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)

        for epsilon in epsilons:
            adv_img = image_batch[0][0].reshape(
                gradient_sign.shape) + epsilon * gradient_sign
            adv_img = np.clip(adv_img, min_, max_)
            adv_label = np.argmax(self.model.predict([(adv_img, 0)]))
            #print("pre_label="+str(pre_label)+ " adv_label="+str(adv_label))
            if pre_label != adv_label:
                #print(epsilon, pre_label, adv_label)
                return adv_img


FGSM = GradientSignAttack
