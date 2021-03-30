# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict
from ...fluid.dygraph.layers import Layer

__all__ = ['LayerDict', ]


class LayerDict(Layer):
    def __init__(self, sublayers=None):
        super(LayerDict, self).__init__()
        if sublayers is not None:
            self._sub_layers.update(sublayers)

    def __getitem__(self, key: str):
        return self._sub_layers[key]

    def __setitem__(self, key: str, sublayer):
        return self.add_sublayer(key, sublayer)

    def __delitem__(self, key: str):
        del self._sub_layers[key]

    def __len__(self):
        return len(self._sub_layers)

    def __iter__(self):
        return iter(self._sub_layers.values())

    def __contains__(self, key: str):
        return key in self._sub_layers

    def clear(self):
        """
        """
        self._sub_layers.clear()

    def pop(self, key: str):
        """
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        """
        """
        return self._sub_layers.keys()

    def items(self):
        """
        """
        return self._sub_layers.items()

    def values(self):
        """
        """
        return self._sub_layers.values()

    def update(self, sublayers):
        """
        """
        for key, layer in sublayers.items():
            self.add_sublayer(key, layer)
        return self
