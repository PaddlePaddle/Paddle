# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import types
import warnings
from dataclasses import dataclass, field


@dataclass
class LayerSpec:
    """This is a Layer Specification dataclass.

    Specification defines the location of the layer (to import dynamically)
    or the imported layer itself. It also defines the extra_kwargs that need to be
    passed to initialize the layer.

    Args:
        layer (tuple | type): A tuple describing the location of the
            layer class e.g. `(layer.location, LayerClass)` or the imported
            layer class itself e.g. `LayerClass` (which is already imported
            using `from layer.location import LayerClass`).
        extra_kwargs (dict): A dictionary of extra_kwargs that need to be passed while init.

    """

    layer: tuple | type
    extra_kwargs: dict = field(default_factory=lambda: {})
    sublayers_spec: type = None

    def __repr__(self):
        rst = ""
        if isinstance(self.layer, tuple):
            for sub_layer in self.layer:
                rst = rst + repr(sub_layer) + ","
        else:
            rst = repr(self.layer) + repr(self.extra_kwargs)
        return rst


def import_spec_layer(layer_path: tuple[str]):
    """Import a named object from a layer in the context of this function."""
    base_path, name = layer_path
    try:
        layer = __import__(base_path, globals(), locals(), [name])
    except ImportError as e:
        print(f"couldn't import layer due to {e}")
        return None
    return vars(layer)[name]


def get_spec_layer(spec_or_layer: LayerSpec | type, **additional_kwargs):
    # If a layer class is already provided return it as is
    if isinstance(spec_or_layer, (type, types.FunctionType)):
        return spec_or_layer

    # If the layer is provided instead of layer path, then return it as is
    if isinstance(spec_or_layer.layer, (type, types.FunctionType)):
        return spec_or_layer.layer

    # Otherwise, return the dynamically imported layer from the layer path
    return import_spec_layer(spec_or_layer.layer)


def build_spec_layer(spec_or_layer: LayerSpec | type, *args, **kwargs):
    # If the passed `spec_or_layer` is
    # a `Function`, then return it as it is
    # NOTE: to support an already initialized layer add the following condition
    # `or isinstance(spec_or_layer, paddle.nn.Layer)` to the following if check
    if isinstance(spec_or_layer, types.FunctionType):
        return spec_or_layer

    # If the passed `spec_or_layer` is actually a spec (instance of
    # `LayerSpec`) and it specifies a `Function` using its `layer`
    # field, return the `Function` as it is
    if isinstance(spec_or_layer, LayerSpec) and isinstance(
        spec_or_layer.layer, types.FunctionType
    ):
        return spec_or_layer.layer

    # Check if a layer class is provided as a spec or if the layer path
    # itself is a class
    if isinstance(spec_or_layer, type):
        layer = spec_or_layer
    elif hasattr(spec_or_layer, "layer") and isinstance(
        spec_or_layer.layer, type
    ):
        layer = spec_or_layer.layer
    else:
        # Otherwise, dynamically import the layer from the layer path
        layer = import_spec_layer(spec_or_layer.layer)

    # If the imported layer is actually a `Function` return it as it is
    if isinstance(layer, types.FunctionType):
        return layer

    # Finally return the initialized layer with extra_kwargs from the spec as well
    # as those passed as **kwargs from the code

    # Add the `sublayers_spec` argument to the layer init call if it exists in the
    # spec.
    if (
        hasattr(spec_or_layer, "sublayers_spec")
        and spec_or_layer.sublayers_spec is not None
    ):
        kwargs["sublayers_spec"] = spec_or_layer.sublayers_spec
    if hasattr(spec_or_layer, "extra_kwargs"):
        for key in spec_or_layer.extra_kwargs.keys():
            if key in kwargs:
                warnings.warn(
                    f"Got same key {key} in extra_kwargs and kwargs during init {layer.__name__}. Will keep the value ing extra_kwargs."
                )
                kwargs.pop(key)
    try:
        return layer(
            *args,
            **spec_or_layer.extra_kwargs
            if hasattr(spec_or_layer, "extra_kwargs")
            else {},
            **kwargs,
        )
    except Exception as e:
        # improve the error message since we hide the layer name in the line above
        import sys

        raise type(e)(
            f"{e!s} when instantiating {layer.__name__}"
        ).with_traceback(sys.exc_info()[2])
