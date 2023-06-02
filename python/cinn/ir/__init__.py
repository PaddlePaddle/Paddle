# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

from ..core_api.ir import *


def get_global_func(name):
    return Registry.get(name)


def register(name, override=False):
    def _register_fn(fn):
        Registry.register(name, override).set_body(PackedFunc(fn))
        return Registry.get(name)

    return _register_fn


def register_packed_func(name, override=False):
    def _register(fn):
        def _packed(args, rv):
            _args = []
            for i in range(len(args)):
                _args.append(args[i])
            r = fn(*_args)
            rv.set(r)

        Registry.register(name, override).set_body(PackedFunc(_packed))
        return Registry.get(name)

    return _register
