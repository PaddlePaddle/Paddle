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

<<<<<<< HEAD
__all__ = ['ProbabilityEntry', 'CountFilterEntry']


class EntryAttr:
=======
from __future__ import print_function

__all__ = ['ProbabilityEntry', 'CountFilterEntry']


class EntryAttr(object):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    """
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
    """

    def __init__(self):
        self._name = None

    def _to_attr(self):
        """
        Returns the attributes of this parameter.

        Returns:
            Parameter attributes(map): The attributes of this parameter.
        """
        raise NotImplementedError("EntryAttr is base class")


class ProbabilityEntry(EntryAttr):
<<<<<<< HEAD
    def __init__(self, probability):
        super().__init__()
=======

    def __init__(self, probability):
        super(ProbabilityEntry, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if not isinstance(probability, float):
            raise ValueError("probability must be a float in (0,1)")

        if probability <= 0 or probability >= 1:
            raise ValueError("probability must be a float in (0,1)")

        self._name = "probability_entry"
        self._probability = probability

    def _to_attr(self):
        return ":".join([self._name, str(self._probability)])


class CountFilterEntry(EntryAttr):
<<<<<<< HEAD
    def __init__(self, count_filter):
        super().__init__()

        if not isinstance(count_filter, int):
            raise ValueError(
                "count_filter must be a valid integer greater than 0"
            )

        if count_filter < 0:
            raise ValueError(
                "count_filter must be a valid integer greater or equal than 0"
            )
=======

    def __init__(self, count_filter):
        super(CountFilterEntry, self).__init__()

        if not isinstance(count_filter, int):
            raise ValueError(
                "count_filter must be a valid integer greater than 0")

        if count_filter < 0:
            raise ValueError(
                "count_filter must be a valid integer greater or equal than 0")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self._name = "count_filter_entry"
        self._count_filter = count_filter

    def _to_attr(self):
        return ":".join([self._name, str(self._count_filter)])
