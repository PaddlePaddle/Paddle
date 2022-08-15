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

__all__ = ["LazyInit"]


class LazyGuard(object):
    """
    Guard Context to trigger switching mode between dygraph and static mode,
    and holds the startup program resource.
    """

    def __init__(self):
        self._init_program()

        self._state = False
        self._tracer = None
        self._in_guard = False

    def enable(self, clear_cache=True):
        """
        Switch into lazy mode.

        NOTE(dev): This is a very low level API and not exposed for user.
        """
        if self._state:
            return
        assert framework.in_dygraph_mode(
        ), "LazyInit.enable() is only available in dygraph mode."
        self._state = True

        if clear_cache:
            self._init_program()

    def disable(self):
        """
        Exit from lazy mode.

        NOTE(dev): This is a very low level API and not exposed for user.
        """
        if not self._state:
            return
        self._state = False

    def _init_program(self):
        self.startup_program = framework.Program()

    def __enter__(self):
        """
        Switch into lazy mode and set _dygraph_tracer_ with None to convert
        dygraph mode into static mode.
        """
        self.enable(clear_cache=True)
        if self._in_guard: return
        self._tracer = framework._dygraph_tracer_
        framework._dygraph_tracer_ = None
        self._in_guard = True

    def __exit__(self, *args, **kwargs):
        """
        Exit from lazy mode and recover _dygraph_tracer_.
        """
        self.disable()
        if not self._in_guard: return
        assert self._tracer is not None
        framework._dygraph_tracer_ = self._tracer
        self._tracer = None
        self._in_guard = False

    @property
    def state(self):
        return self._state


_lazy_guard = LazyGuard()


def lazy_guard():
    global _lazy_guard
    return _lazy_guard


class LazyInit(object):
    """
    LazyInit is a wrapper interface for nn.Layer, it forwards the construct
    process of user defined Layer. Meanwhile, it provides necessary API to
    trigger EagerParamBase Lazy Initialization and get startup Program.
    """

    def __init__(self, class_obj=None):
        self.class_obj = class_obj
        self.clear_cache = True

    def __call__(self, *args, **kwargs):
        """
        Construct instance from class_obj by Lazy Initializing parameters.

        Examples:
    
            .. code-block:: python

                from paddle import LazyInit
                from paddle.nn import Linear
                
                fc = LazyInit(Linear)(10, 10)

                for param in fc.parameters():
                    param.initialize()
        """
        assert issubclass(
            self.class_obj, type
        ), "Required class_obj must be a class type, but received %s." % self.class_obj
        assert isinstance()
        global _lazy_guard
        _lazy_guard.enable(self.clear_cache)
        # construct Layer instance
        with framework.program_guard(framework.Program()):
            instance = self.class_obj(*args, **kwargs)
        _lazy_guard.disable()
        # set @property dynamically to visit startup_program
        instance.startup_program = _lazy_guard.startup_program

        return instance

    def __enter__(self):
        """
        Context manager to support 'with' keyword.

        Examples:
    
            .. code-block:: python

                from paddle import LazyInit
                from paddle.nn import Linear

                with LazyInit() as lz:
                    fc = Linear(10, 10)

                    print(lz.startup_program())
                
                # outer 'with'
                for param in fc.parameters():
                    param.initialize()
        """
        _lazy_guard.enable()
        return self

    def __exit__(self, *args, **kwargs):
        _lazy_guard.disable()

    @staticmethod
    def startup_program():
        """
        A static method to get startup program for the latest Layer.

        Examples:
    
            .. code-block:: python

                from paddle import LazyInit
                from paddle.nn import Linear
                
                fc = LazyInit(Linear)(10, 10)

                print(LazyInit.startup_program())
            
        """
        return _lazy_guard.startup_program
