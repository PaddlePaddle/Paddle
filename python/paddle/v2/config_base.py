# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import collections
import re
import paddle.trainer_config_helpers as conf_helps

__layer_map__ = {}


def __map_docstr__(doc, name):
    if doc is None:
        return doc

    assert isinstance(doc, basestring)

    # replace LayerOutput to paddle.v2.config_base.Layer
    doc = doc.replace("LayerOutput", "paddle.v2.config_base.Layer")

    doc = doc.replace('ParameterAttribute', 'paddle.v2.attr.ParameterAttribute')

    doc = re.sub(r'ExtraLayerAttribute[^\s]?', 'paddle.v2.attr.ExtraAttribute',
                 doc)

    # xxx_layer to xxx
    doc = re.sub(r"(?P<name>[a-z]+)_layer", r"\g<name>", doc)

    # XxxxActivation to paddle.v2.activation.Xxxx
    doc = re.sub(r"(?P<name>[A-Z][a-zA-Z]+)Activation",
                 r"paddle.v2.activation.\g<name>", doc)

    # xxx_evaluator to paddle.v2.evaluator.xxx
    doc = re.sub(r"(?P<name>[a-z]+)_evaluator", r"evaluator.\g<name>", doc)

    # TODO(yuyang18): Add more rules if needed.
    return doc


def __convert_to_v2__(f, name, module):
    def wrapped(*args, **xargs):
        out = f(*args, **xargs)
        outs = out
        if not isinstance(out, collections.Sequence):
            outs = [out]
        for l in outs:
            if isinstance(l, conf_helps.LayerOutput):
                __layer_map__[l.full_name] = l
        return out

    wrapped.__doc__ = __map_docstr__(f.__doc__, name)
    wrapped.__name__ = name
    wrapped.__module__ = module

    return wrapped


Layer = conf_helps.LayerOutput
