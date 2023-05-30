# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import paddle

from .infer_meta import MetaInfo
from .utils import NameGenerator, Singleton, log


# global variables
@Singleton
class ProxyTensorContext:
    def __init__(self):
        self.reset()

    def reset(self):
        self.runtime_name_to_proxy_tensor: dict[str, ProxyTensor] = {}
        self.runtime_proxy_tensor_to_name: dict[int, str] = {}
        self.tensor_to_proxy_tensor: dict[int, ProxyTensor] = {}
        self.var_name_generator = NameGenerator("var_")

    def new_varname(self):
        return self.var_name_generator.next()

    def from_tensor(self, tensor) -> ProxyTensor:
        # TODO: don't have the same name.
        if self.tensor_to_proxy_tensor.get(id(tensor), None) is not None:
            return self.tensor_to_proxy_tensor[id(tensor)]

        # TODO(id may have collision)
        name = self.new_varname()
        proxy_tensor = ProxyTensor(name, MetaInfo.from_tensor(tensor))
        self.tensor_to_proxy_tensor[id(tensor)] = proxy_tensor
        proxy_tensor.set_value(tensor)
        return proxy_tensor

    def bind_name_to_proxy_tensor(self, name, proxy_tensor):
        self.runtime_name_to_proxy_tensor[name] = proxy_tensor
        self.runtime_proxy_tensor_to_name[id(proxy_tensor)] = name

    def clear_proxy_tensor_by_name(self, name):
        log(3, f"[GC] trying to GC {name}\n")
        proxy_tensor = self.runtime_name_to_proxy_tensor[name]
        proxy_tensor_id = id(proxy_tensor)
        has_value = proxy_tensor.value() is not None
        eager_tensor_id = id(proxy_tensor.value())

        del self.runtime_name_to_proxy_tensor[name]
        del self.runtime_proxy_tensor_to_name[proxy_tensor_id]
        if has_value and eager_tensor_id in self.tensor_to_proxy_tensor:
            del self.tensor_to_proxy_tensor[eager_tensor_id]
        log(3, f"[GC] {name} GCed\n")

    def get_runtime(self):
        return self.runtime_name_to_proxy_tensor


class ProxyTensor:
    def __init__(self, name, meta):
        self.name: str = name
        self.meta: MetaInfo = meta
        self.value_: paddle.Tensor = None
        ProxyTensorContext().bind_name_to_proxy_tensor(name, self)

    @property
    def shape(self):
        # TODO(xiongkun) consider dynamic shape.
        return self.meta.shape

    @property
    def ndim(self):
        return len(self.meta.shape)

    @property
    def dtype(self):
        return self.meta.dtype

    def set_value(self, value):
        """
        value is a eager tensor.
        when a proxytensor have value, it means it can be evaluated outer to_static.
        """
        self.value_ = value

    def clear_value(self):
        self.value_ = None

    def value(self):
        return self.value_
