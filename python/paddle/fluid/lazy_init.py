# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from . import framework

__all__ = ["LazyGuard"]


<<<<<<< HEAD
class LazyInitHelper:
    """
    A Helper Context to trigger switching mode between dygraph and static graph mode,
=======
class LazyInitHelper(object):
    """
    A Helper Context to trigger switching mode between dygraph and static mode,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    and holds the startup program resource.
    """

    def __init__(self):
        self._state = False
        self._tracer = None
        self._in_guard = False

    def enable(self):
        """
        Switch into lazy mode.

        NOTE(dev): This is a very low level API and not exposed for user.
        """
        if self._state:
            return
<<<<<<< HEAD
        assert (
            framework._non_static_mode()
=======
        assert framework._non_static_mode(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ), "LazyInit.enable() is only available in dygraph mode."
        self._state = True

    def disable(self):
        """
        Exit from lazy mode.

        NOTE(dev): This is a very low level API and not exposed for user.
        """
        if not self._state:
            return
        self._state = False

    def __enter__(self):
        """
        Switch into lazy mode and set _dygraph_tracer_ with None to convert
<<<<<<< HEAD
        dygraph mode into static graph mode.
        """
        self.enable()
        if self._in_guard:
            return
        self._tracer = framework.global_var._dygraph_tracer_
        framework.global_var._dygraph_tracer_ = None
=======
        dygraph mode into static mode.
        """
        self.enable()
        if self._in_guard: return
        self._tracer = framework._dygraph_tracer_
        framework._dygraph_tracer_ = None
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._in_guard = True

    def __exit__(self, *args, **kwargs):
        """
        Exit from lazy mode and recover _dygraph_tracer_.
        """
        self.disable()
<<<<<<< HEAD
        if not self._in_guard:
            return
        assert self._tracer is not None
        framework.global_var._dygraph_tracer_ = self._tracer
=======
        if not self._in_guard: return
        assert self._tracer is not None
        framework._dygraph_tracer_ = self._tracer
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._tracer = None
        self._in_guard = False

    @property
    def state(self):
        return self._state


_lazy_init_helper = LazyInitHelper()


def lazy_init_helper():
    global _lazy_init_helper
    return _lazy_init_helper


<<<<<<< HEAD
class LazyGuard:
=======
class LazyGuard(object):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    """
    LazyGuard is a wrapper interface for nn.Layer, it forwards the construct
    process of user defined Layer. Meanwhile, it provides necessary API to
    trigger EagerParamBase Lazy Initialization and get startup Program.
    """

    def __enter__(self):
        """
        Construct instance from class_obj by Lazy Initializing parameters.

        Examples:
<<<<<<< HEAD

=======
    
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            .. code-block:: python

                from paddle import LazyGuard
                from paddle.nn import Linear
<<<<<<< HEAD

=======
                
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                with LazyGuard():
                    fc = LazyInit(Linear)(10, 10)

                for param in fc.parameters():
                    param.initialize()
        """
        lazy_init_helper().enable()

    def __exit__(self, *args, **kwargs):
        lazy_init_helper().disable()
