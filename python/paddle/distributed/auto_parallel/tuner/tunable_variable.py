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

import numpy as np


<<<<<<< HEAD
class TunableVariable:
=======
class TunableVariable(object):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    """
    Tunablevariable base class.
    """

    def __init__(self, name, default=None):
        self.name = name
        self._default = default

    @property
    def default(self):
        return self._default

    def get_state(self):
        return {"name": self.name, "default": self.default}

    @classmethod
    def from_state(cls, state):
        return cls(**state)


class Fixed(TunableVariable):
    """
    Fixed variable which cannot be changed.
    """

    def __init__(self, name, default):
<<<<<<< HEAD
        super().__init__(name=name, default=default)
=======
        super(Fixed, self).__init__(name=name, default=default)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.name = name
        if not isinstance(default, (str, int, float, bool)):
            raise ValueError(
                "Fixed must be an str, int, float or bool, but found {}".format(
<<<<<<< HEAD
                    default
                )
            )
=======
                    default))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._default = default

    def random(self, seed=None):
        return self._default

    def __repr__(self):
        return "Fixed(name: {}, value: {})".format(self.name, self.default)


class Boolean(TunableVariable):
    """
    Choice between True and False.
    """

    def __init__(self, name, default=False):
<<<<<<< HEAD
        super().__init__(name=name, default=default)
        if default not in {True, False}:
            raise ValueError(
                "default must be a Python boolean, but got {}".format(default)
            )
=======
        super(Boolean, self).__init__(name=name, default=default)
        if default not in {True, False}:
            raise ValueError(
                "default must be a Python boolean, but got {}".format(default))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def random(self, seed=None):
        rng = np.random.default_rng(seed)
        return rng.choice((True, False))

    def __repr__(self):
        return 'Boolean(name: "{}", default: {})'.format(
<<<<<<< HEAD
            self.name, self.default
        )


class Choice(TunableVariable):
    def __init__(self, name, values, default=None):
        super().__init__(name=name, default=default)
=======
            self.name, self.default)


class Choice(TunableVariable):

    def __init__(self, name, values, default=None):
        super(Choice, self).__init__(name=name, default=default)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        types = set(type(v) for v in values)
        if len(types) > 1:
            raise TypeError(
<<<<<<< HEAD
                "Choice can contain only one type of value, but found values: {} with types: {}.".format(
                    str(values), str(types)
                )
            )
=======
                "Choice can contain only one type of value, but found values: {} with types: {}."
                .format(str(values), str(types)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._is_unknown_type = False

        if isinstance(values[0], str):
            values = [str(v) for v in values]
            if default is not None:
                default = str(default)
        elif isinstance(values[0], int):
            values = [int(v) for v in values]
            if default is not None:
                default = int(default)
        elif isinstance(values[0], float):
            values = [float(v) for v in values]
            if default is not None:
                default = float(default)
        elif isinstance(values[0], bool):
            values = [bool(v) for v in values]
            if default is not None:
                default = bool(default)
        else:
            self._is_unknown_type = True
            self._indices = [i for i in range(len(values))]
        self.values = values

        if default is not None and default not in values:
            raise ValueError(
<<<<<<< HEAD
                "The default value should be one of the choices {}, but found {}".format(
                    values, default
                )
            )
=======
                "The default value should be one of the choices {}, but found {}"
                .format(values, default))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._default = default

    @property
    def default(self):
        if self._default is None:
            if None in self.values:
                return None
            return self.values[0]
        return self._default

    def random(self, seed=None):
        rng = np.random.default_rng(seed)
        if self._is_unknown_type:
            indice = rng.choice(self._indices)
            return self.values[indice]
        else:
            return rng.choice(self.values)

    def get_state(self):
<<<<<<< HEAD
        state = super().get_state()
=======
        state = super(Choice, self).get_state()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        state["values"] = self.values
        return state

    def __repr__(self):
        return 'Choice(name: "{}", values: {}, default: {})'.format(
<<<<<<< HEAD
            self.name, self.values, self.default
        )
=======
            self.name, self.values, self.default)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


class IntRange(TunableVariable):
    """
    Integer range.
    """

    def __init__(self, name, start, stop, step=1, default=None, endpoint=False):
<<<<<<< HEAD
        super().__init__(name=name, default=default)
=======
        super(IntRange, self).__init__(name=name, default=default)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.start = self._check_int(start)
        self.stop = self._check_int(stop)
        self.step = self._check_int(step)
        self._default = default
        self.endpoint = endpoint

    @property
    def default(self):
        if self._default is not None:
            return self._default
        return self.start

    def random(self, seed=None):
        rng = np.random.default_rng(seed)
        value = (self.stop - self.start) * rng.random() + self.start
        if self.step is not None:
            if self.endpoint:
                values = np.arange(self.start, self.stop + 1e-7, step=self.step)
            else:
                values = np.arange(self.start, self.stop, step=self.step)
            closest_index = np.abs(values - value).argmin()
            value = values[closest_index]
        return int(value)

    def get_state(self):
<<<<<<< HEAD
        state = super().get_state()
=======
        state = super(IntRange, self).get_state()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        state["start"] = self.start
        state["stop"] = self.stop
        state["step"] = self.step
        state["default"] = self._default
        return state

    def _check_int(self, val):
        int_val = int(val)
        if int_val != val:
<<<<<<< HEAD
            raise ValueError(
                "Expects val is an int, but found: {}.".format(str(val))
            )
=======
            raise ValueError("Expects val is an int, but found: {}.".format(
                str(val)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return int_val

    def __repr__(self):
        return "IntRange(name: {}, start: {}, stop: {}, step: {}, default: {})".format(
<<<<<<< HEAD
            self.name, self.start, self.stop, self.step, self.default
        )
=======
            self.name, self.start, self.stop, self.step, self.default)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


class FloatRange(TunableVariable):
    """
    Float range.
    """

<<<<<<< HEAD
    def __init__(
        self, name, start, stop, step=None, default=None, endpoint=False
    ):
        super().__init__(name=name, default=default)
=======
    def __init__(self,
                 name,
                 start,
                 stop,
                 step=None,
                 default=None,
                 endpoint=False):
        super(FloatRange, self).__init__(name=name, default=default)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.stop = float(stop)
        self.start = float(start)
        if step is not None:
            self.step = float(step)
        else:
            self.step = None
        self._default = default
        self.endpoint = endpoint

    @property
    def default(self):
        if self._default is not None:
            return self._default
        return self.start

    def random(self, seed=None):
        rng = np.random.default_rng(seed)
        value = (self.stop - self.start) * rng.random() + self.start
        if self.step is not None:
            if self.endpoint:
                values = np.arange(self.start, self.stop + 1e-7, step=self.step)
            else:
                values = np.arange(self.start, self.stop, step=self.step)
            closest_index = np.abs(values - value).argmin()
            value = values[closest_index]
        return value

    def get_state(self):
<<<<<<< HEAD
        state = super().get_state()
=======
        state = super(FloatRange, self).get_state()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        state["start"] = self.start
        state["stop"] = self.stop
        state["step"] = self.step
        state["endpoint"] = self.endpoint
        return state

    def __repr__(self):
        return "FloatRange(name: {}, start: {}, stop: {}, step: {}, default: {}, endpoint: {})".format(
<<<<<<< HEAD
            self.name,
            self.start,
            self.stop,
            self.step,
            self.default,
            self.endpoint,
        )
=======
            self.name, self.start, self.stop, self.step, self.default,
            self.endpoint)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
