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

from collections import OrderedDict
from typing import Dict

import paddle
from paddle import nn


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function ensures that all layers have a channel number that is divisible by divisor
    You can also see at https://github.com/keras-team/keras/blob/8ecef127f70db723c158dbe9ed3268b3d610ab55/keras/applications/mobilenet_v2.py#L505

    Args:
        divisor (int): The divisor for number of channels. Default: 8.
        min_value (int, optional): The minimum value of number of channels, if it is None,
                the default is divisor. Default: None.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class IntermediateLayerGetter(nn.LayerDict):
    """
    Layer wrapper that returns intermediate layers from a model.

    It has a strong assumption that the layers have been registered into the model in the
    same order as they are used. This means that one should **not** reuse the same nn.Layer
    twice in the forward if you want this to work.

    Additionally, it is only able to query sublayer that are directly assigned to the model.
    So if `model` is passed, `model.feature1` can be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Layer): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names of the layers for
        which the activations will be returned as the key of the dict, and the value of the
        dict is the name of the returned activation (which the user can specify).

    Examples:
        .. code-block:: python

        import paddle
        m = paddle.vision.models.resnet18(pretrained=False)
        # extract layer1 and layer3, giving as names `feat1` and feat2`
        new_m = paddle.vision.models._utils.IntermediateLayerGetter(m,
            {'layer1': 'feat1', 'layer3': 'feat2'})
        out = new_m(paddle.rand([1, 3, 224, 224]))
        print([(k, v.shape) for k, v in out.items()])
        # [('feat1', [1, 64, 56, 56]), ('feat2', [1, 256, 14, 14])]
    """

    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Layer, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():

            if (isinstance(module, nn.Linear) and x.ndim == 4) or (
                len(module.sublayers()) > 0
                and isinstance(module.sublayers()[0], nn.Linear)
                and x.ndim == 4
            ):
                x = paddle.flatten(x, 1)

            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
