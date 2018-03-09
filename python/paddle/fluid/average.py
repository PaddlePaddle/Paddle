#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
"""
    Class of all kinds of Average.

    All Averages are accomplished via Python totally. 
    They do not change Paddle's Program, nor do anything to
    modify NN model's configuration. They are completely 
    wrappers of Python functions.
"""


def _is_number_(var):
    return isinstance(var, int) or isinstance(var, float) or (isinstance(
        var, np.ndarray) and var.shape == (1, ))


def _is_number_or_matrix_(var):
    return _is_number_(var) or isinstance(var, np.ndarray)


class WeightedAverage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.numerator = None
        self.denominator = None

    def add(self, value, weight):
        if not _is_number_or_matrix_(value):
            raise ValueError(
                "The 'value' must be a number(int, float) or a numpy ndarray.")
        if not _is_number_(weight):
            raise ValueError("The 'weight' must be a number(int, float).")

        if self.numerator is None or self.denominator is None:
            self.numerator = value * weight
            self.denominator = weight
        else:
            self.numerator += value * weight
            self.denominator += weight

    def eval(self):
        if self.numerator is None or self.denominator is None:
            raise ValueError(
                "There is no data to be averaged in WeightedAverage.")
        return self.numerator / self.denominator
