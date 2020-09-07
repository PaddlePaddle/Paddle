# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
import numpy as np
import numbers

import paddle
import paddle.nn as nn
from paddle.static import InputSpec

from collections import OrderedDict

__all__ = ['summary']


def summary(net, input_size, batch_size=None, dtypes=None):
    """Prints a string summary of the network.

    Args:
        net (Layer): the network which must be a subinstance of Layer.
        input_size (tuple|InputSpec|list[tuple|InputSpec]): size of input tensor. if model only 
                    have one input, input_size can be tuple or InputSpec. if model
                    have multiple input, input_size must be a list which contain 
                    every input's shape.
        batch_size (int, optional): batch size of input tensor, Default: None.
        dtypes (str, optional): if dtypes is None, 'float32' will be used, Default: None.

    Returns:
        Dict: a summary of the network including total params and total trainable params.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            class LeNet(nn.Layer):
                def __init__(self, num_classes=10):
                    super(LeNet, self).__init__()
                    self.num_classes = num_classes
                    self.features = nn.Sequential(
                        nn.Conv2d(
                            1, 6, 3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(
                            6, 16, 5, stride=1, padding=0),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2))

                    if num_classes > 0:
                        self.fc = nn.Sequential(
                            nn.Linear(400, 120),
                            nn.Linear(120, 84),
                            nn.Linear(
                                84, 10))

                def forward(self, inputs):
                    x = self.features(inputs)

                    if self.num_classes > 0:
                        x = paddle.flatten(x, 1)
                        x = self.fc(x)
                    return x

            lenet = LeNet()

            params_info = paddle.summary(lenet, (1, 28, 28))
            print(params_info)

    """
    if isinstance(input_size, InputSpec):
        _input_size = tuple(input_size.shape[1:])
        if batch_size is None:
            batch_size = input_size.shape[0]
    elif isinstance(input_size, list):
        _input_size = []
        for item in input_size:
            if isinstance(item, int):
                item = (item, )
            assert isinstance(item,
                              (tuple, InputSpec)), 'When input_size is list, \
            expect item in input_size is a tuple or InputSpec, but got {}'.format(
                                  type(item))

            if isinstance(item, InputSpec):
                _input_size.append(tuple(item.shape[1:]))
                if batch_size is None:
                    batch_size = item.shape[0]
            else:
                _input_size.append(item)
    elif isinstance(input_size, int):
        _input_size = (input_size, )
    else:
        _input_size = input_size

    if batch_size is None:
        batch_size = -1

    if not paddle.in_dynamic_mode():
        warnings.warn(
            "Your model was created in static mode, this may not get correct summary information!"
        )

    result, params_info = summary_string(net, _input_size, batch_size, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, dtypes=None):
    if dtypes == None:
        dtypes = ['float32'] * len(input_size)

    summary_str = ''

    depth = len(list(model.sublayers()))

    def register_hook(layer):
        def hook(layer, input, output):
            class_name = str(layer.__class__).split(".")[-1].split("'")[0]

            try:
                layer_idx = int(layer._full_name.split('_')[-1])
            except:
                layer_idx = len(summary)

            m_key = "%s-%i" % (class_name, layer_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].shape)
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.shape)[1:]
                                                  for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.shape)
                summary[m_key]["output_shape"][0] = batch_size

            params = 0

            if paddle.in_dynamic_mode():
                layer_state_dict = layer._parameters
            else:
                layer_state_dict = layer.state_dict()

            for k, v in layer_state_dict.items():
                params += np.prod(v.shape)

                try:
                    if (getattr(getattr(layer, k), 'trainable')) and (
                            not getattr(getattr(layer, k), 'stop_gradient')):
                        summary[m_key]["trainable"] = True
                    else:
                        summary[m_key]["trainable"] = False
                except:
                    summary[m_key]["trainable"] = True

            summary[m_key]["nb_params"] = params

        if (not isinstance(layer, nn.Sequential) and
                not isinstance(layer, nn.LayerList) and
            (not (layer == model) or depth < 1)):

            hooks.append(layer.register_forward_post_hook(hook))

    def _check_input_size(input_sizes):
        for input_size in input_sizes:
            for item in input_size:
                if not isinstance(item, numbers.Number):
                    raise TypeError(
                        "Expected item in input size be a number, but got {}".
                        format(type(item)))

                if item <= 0:
                    raise ValueError(
                        "Expected item in input size greater than zero, but got {}".
                        format(item))

    if isinstance(input_size, tuple):
        input_size = [input_size]

    _check_input_size(input_size)

    x = [
        paddle.rand(
            [2] + list(in_size), dtype=dtype)
        for in_size, dtype in zip(input_size, dtypes)
    ]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    table_width = 80
    summary_str += "-" * table_width + "\n"
    line_new = "{:>15} {:>20} {:>20} {:>15}".format(
        "Layer (type)", "Input Shape", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "=" * table_width + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>15} {:>20} {:>20} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]), )
        total_params += summary[layer]["nb_params"]

        try:
            total_output += np.prod(summary[layer]["output_shape"])
        except:
            for output_shape in summary[layer]["output_shape"]:
                total_output += np.prod(output_shape)

        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(
        np.prod(sum(input_size, ())) * batch_size * 4. / (1024**2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024**2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024**2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "=" * table_width + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "-" * table_width + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "-" * table_width + "\n"
    # return summary
    return summary_str, {
        'total_params': total_params,
        'trainable_params': trainable_params
    }
