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


class TunableVariable(object):
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


class Boolean(TunableVariable):
    """
    Choice between True and False.
    """

    def __init__(self, name, default=False, **kwargs):
        super(Boolean, self).__init__(name=name, default=default, **kwargs)
        if default not in {True, False}:
            raise ValueError("`default` must be a Python boolean, but got {}".
                             format(default))

    def __repr__(self):
        return 'Boolean(name: "{}", default: {})'.format(self.name,
                                                         self.default)


class Fixed(TunableVariable):
    """
    Fixed variable which cannot be changed.
    """

    def __init__(self, name, value, **kwargs):
        super(Fixed, self).__init__(name=name, default=value, **kwargs)
        self.name = name
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                "`Fixed` must be an `str`, `int`, `float` or `bool`, but found {}"
                .format(value))
        self.value = value

    @property
    def default(self):
        return self.value

    def get_state(self):
        state = super(Fixed, self).get_state()
        return state

    def __repr__(self):
        return "Fixed(name: {}, value: {})".format(self.name, self.value)


class Choice(TunableVariable):
    def __init__(self, name, values, default=None, ordered=None, **kwargs):
        super(Choice, self).__init__(name=name, default=default, **kwargs)

        types = set(type(v) for v in values)
        if len(types) > 1:
            raise TypeError(
                "`Choice` can contain only one type of value, but found values: {} with types: {}."
                .format(str(values), str(types)))

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
            raise TypeError(
                "`Choice` can only contain `str`, `int`, `float`, or `boll`, but found values: {} "
                .format(str(values)))
        self.values = values

        if default is not None and default not in values:
            raise ValueError(
                "The default value should be one of the choices {}, but found {}".
                format(values, default))
        self._default = default

        self.ordered = ordered
        is_numerical = isinstance(values[0], (int, float))
        if self.ordered and not is_numerical:
            raise ValueError("`ordered` must be `False` for non-numerical "
                             "types.")
        if self.ordered is None:
            self.ordered = is_numerical

    @property
    def default(self):
        if self._default is None:
            if None in self.values:
                return None
            return self.values[0]
        return self._default

    def get_state(self):
        state = super(Choice, self).get_state()
        state["values"] = self.values
        state["ordered"] = self.ordered
        return state

    def __repr__(self):
        return 'Choice(name: "{}", values: {}, default: {}, ordered: {})'.format(
            self.name, self.values, self.ordered, self.default)


class IntRange(TunableVariable):
    """
    Integer range.
    """

    def __init__(self, name, start, end, step=1, default=None, **kwargs):
        super(IntRange, self).__init__(name=name, default=default, **kwargs)
        self.end = self._check_int(end, arg="end")
        self.start = self._check_int(start, arg="start")
        self.step = self._check_int(step, arg="step")

    @property
    def default(self):
        if self._default is not None:
            return self._default
        return self.start

    def get_state(self):
        state = super(IntRange, self).get_state()
        state["start"] = self.start
        state["end"] = self.end
        state["step"] = self.step
        state["default"] = self._default
        return state

    def _check_int(self, val, arg):
        int_val = int(val)
        if int_val != val:
            raise ValueError("Expects val is an int, but found: {}.".format(
                str(val)))
        return int_val

    def __repr__(self):
        return "IntRange(name: {}, start: {}, end: {}, step: {}, default: {})".format(
            self.name, self.start, self.end, self.step, self.default)


class FloatRange(TunableVariable):
    """
    Float range.
    """

    def __init__(self, name, start, end, step=None, default=None, **kwargs):
        super(FloatRange, self).__init__(name=name, default=default, **kwargs)
        self.end = float(end)
        self.start = float(start)
        if step is not None:
            self.step = float(step)
        else:
            self.step = None

    @property
    def default(self):
        if self._default is not None:
            return self._default
        return self.start

    def get_state(self):
        state = super(FloatRange, self).get_state()
        state["start"] = self.start
        state["end"] = self.end
        state["step"] = self.step
        return state

    def __repr__(self):
        return "FloatRange(name: {}, start: {}, end: {}, step: {}, default: {})".format(
            self.name,
            self.start,
            self.end,
            self.step,
            self.default, )
