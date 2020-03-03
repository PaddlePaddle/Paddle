# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import

import six
import abc

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

__all__ = ['Metric']


@six.add_metaclass(abc.ABCMeta)
class Metric(object):
    """
    Base class for metric, encapsulates metric logic and APIs

    Usage:
    m = SomeMetric()
    for prediction, label in ...:
        m.update(prediction, label)
    m.accumulate()
    """

    @abc.abstractmethod
    def reset(self):
        """
        Reset states and result
        """
        raise NotImplementedError("function 'reset' not implemented in {}.".format(self.__class__.__name__))

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """
        Update states for metric
        """
        raise NotImplementedError("function 'update' not implemented in {}.".format(self.__class__.__name__))

    @abc.abstractmethod
    def accumulate(self):
        """
        Accumulates statistics, computes and returns the metric value
        """
        raise NotImplementedError("function 'accumulate' not implemented in {}.".format(self.__class__.__name__))

