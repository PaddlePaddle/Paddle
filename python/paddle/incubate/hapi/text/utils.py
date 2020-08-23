#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import functools
import paddle


class InitTrackerMeta(type(paddle.fluid.dygraph.Layer)):
    """
    Since InitTrackerMeta would be used as metaclass for model, thus use
    type(Layer) rather than type to avoid multiple inheritance metaclass
    conflicts temporarily.
    
    """

    # def __new__(cls, name, bases, attrs):
    #     if '__init__' in attrs:
    #         init_func = attrs['__init__']
    #         help_func = attrs.get('_wrap_init', None)
    #         attrs['__init__'] = cls.wrap_with_conf_tracker(
    #             init_func, help_func)
    #     return type.__new__(cls, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        init_func = cls.__init__
        # If attrs has `__init__`, wrap it using accessable `_wrap_init`.
        # Otherwise, no need to wrap again since the super cls has been wraped.
        # TODO: remove reduplicated tracker if using super cls `__init__`
        help_func = getattr(cls, '_wrap_init',
                            None) if '__init__' in attrs else None
        cls.__init__ = InitTrackerMeta.init_then_track_conf(init_func,
                                                            help_func)
        super(InitTrackerMeta, cls).__init__(name, bases, attrs)

    @staticmethod
    def init_then_track_conf(init_func, help_func=None):
        @functools.wraps(init_func)
        def __impl__(self, *args, **kwargs):
            args_bak = copy.deepcopy(args)
            kwargs_bak = copy.deepcopy(kwargs)
            init_func(self, *args, **kwargs)
            # TODO: Add class info into config
            # any need to use inspect.getfullargspec to rearrange
            if args_bak:
                kwargs_bak['init_inputs'] = args_bak
            self.init_config = kwargs_bak
            # registed helper by `_wrap_init`
            if help_func:
                help_func(self, init_func, *args, **kwargs)

        return __impl__
