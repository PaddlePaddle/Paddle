#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# Notice that the following codes are modified from KerasTuner to implement our own tuner.
# Please refer to https://github.com/keras-team/keras-tuner/blob/master/keras_tuner/engine/hyperparameters.py.

from .tunable_variable import Boolean
from .tunable_variable import Fixed
from .tunable_variable import Choice
from .tunable_variable import IntRange
from .tunable_variable import FloatRange


class TunableSpace:
    """
    A TunableSpace is constructed by the tunable variables.
    """

    def __init__(self):
        # Tunable variables for this tunable variables
        self._variables = {}
        # Specific values coresponding to each tunable variable
        self._values = {}

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, variables):
        self._variables = variables

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values

    def get_value(self, name):
        if name in self.values:
            return self.values[name]
        else:
            raise KeyError("{} does not exist.".format(name))

    def set_value(self, name, value):
        if name in self.values:
            self.values[name] = value
        else:
            raise KeyError("{} does not exist.".format(name))

    def _exists(self, name):
        if name in self._variables:
            return True
        return False

    def _retrieve(self, tv):
        tv = tv.__class__.from_state(tv.get_state())
        if self._exists(tv.name):
            return self.get_value(tv.name)
        return self._register(tv)

    def _register(self, tv):
        self._variables[tv.name] = tv
        if tv.name not in self.values:
            self.values[tv.name] = tv.default
        return self.values[tv.name]

    def __getitem__(self, name):
        return self.get_value(name)

    def __setitem__(self, name, value):
        self.set_value(name, value)

    def __contains__(self, name):
        try:
            self.get_value(name)
            return True
        except (KeyError, ValueError):
            return False

    def fixed(self, name, default):
        tv = Fixed(name=name, default=default)
        return self._retrieve(tv)

    def boolean(self, name, default=False):
        tv = Boolean(name=name, default=default)
        return self._retrieve(tv)

    def choice(self, name, values, default=None):
        tv = Choice(name=name, values=values, default=default)
        return self._retrieve(tv)

    def int_range(self, name, start, stop, step=1, default=None):
        tv = IntRange(
            name=name, start=start, stop=stop, step=step, default=default
        )
        return self._retrieve(tv)

    def float_range(self, name, start, stop, step=None, default=None):
        tv = FloatRange(
            name=name, start=start, stop=stop, step=step, default=default
        )
        return self._retrieve(tv)

    def get_state(self):
        return {
            "variables": [
                {"class_name": v.__class__.__name__, "state": v.get_state()}
                for v in self._variables.values()
            ],
            "values": dict((k, v) for (k, v) in self.values.items()),
        }

    @classmethod
    def from_state(cls, state):
        ts = cls()
        for v in state["variables"]:
            v = _deserialize_tunable_variable(v)
            ts._variables[v.name] = v
        ts._values = dict((k, v) for (k, v) in state["values"].items())
        return ts


def _deserialize_tunable_variable(state):
    classes = (Boolean, Fixed, Choice, IntRange, FloatRange)
    cls_name_to_cls = {cls.__name__: cls for cls in classes}

    if isinstance(state, classes):
        return state

    if (
        not isinstance(state, dict)
        or "class_name" not in state
        or "state" not in state
    ):
        raise ValueError(
            "Expect state to be a python dict containing class_name and state as keys, but found {}".format(
                state
            )
        )

    cls_name = state["class_name"]
    cls = cls_name_to_cls[cls_name]
    if cls is None:
        raise ValueError("Unknown class name {}".format(cls_name))

    cls_state = state["state"]
    deserialized_object = cls.from_state(cls_state)
    return deserialized_object
