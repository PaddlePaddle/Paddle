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

import re
import math
from functools import partial
import collections

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple(
    'GlobalParams',
    [
        'batch_norm_momentum',
        'batch_norm_epsilon',
        'dropout_rate',
        'num_classes',
        'width_coefficient',
        'depth_coefficient',
        'depth_divisor',
        'min_depth',
        'drop_connect_rate',
        'image_size',
    ],
)

# Parameters for an individual model block
BlockArgs = collections.namedtuple(
    'BlockArgs',
    [
        'kernel_size',
        'num_repeat',
        'input_filters',
        'output_filters',
        'expand_ratio',
        'id_skip',
        'stride',
        'se_ratio',
    ],
)

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def round_filters(filters, global_params):
    """Calculate and round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(
        min_depth, int(filters + divisor / 2) // divisor * divisor
    )
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, prob, training):
    """Drop input connection"""
    if not training:
        return inputs
    keep_prob = 1.0 - prob
    inputs_shape = paddle.shape(inputs)
    random_tensor = keep_prob + paddle.rand(shape=[inputs_shape[0], 1, 1, 1])
    binary_tensor = paddle.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
    Static padding is necessary for ONNX exporting of models."""
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2D):
    """2D Convolutions like TensorFlow, for a dynamic image size"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias_attr=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            groups,
            bias_attr=bias_attr,
        )
        self.stride = (
            self._stride if len(self._stride) == 2 else [self._stride[0]] * 2
        )

    def forward(self, x):
        ih, iw = x.shape[-2:]
        kh, kw = self.weight.shape[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max(
            (oh - 1) * self.stride[0] + (kh - 1) * self._dilation[0] + 1 - ih, 0
        )
        pad_w = max(
            (ow - 1) * self.stride[1] + (kw - 1) * self._dilation[1] + 1 - iw, 0
        )
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                ],
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self._padding,
            self._dilation,
            self._groups,
        )


class Conv2dStaticSamePadding(nn.Conv2D):
    """2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(
        self, in_channels, out_channels, kernel_size, image_size=None, **kwargs
    ):
        if 'stride' in kwargs and isinstance(kwargs['stride'], list):
            kwargs['stride'] = kwargs['stride'][0]
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = (
            self._stride if len(self._stride) == 2 else [self._stride[0]] * 2
        )

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (
            image_size if type(image_size) == list else [image_size, image_size]
        )
        kh, kw = self.weight.shape[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max(
            (oh - 1) * self.stride[0] + (kh - 1) * self._dilation[0] + 1 - ih, 0
        )
        pad_w = max(
            (ow - 1) * self.stride[1] + (kw - 1) * self._dilation[1] + 1 - iw, 0
        )
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.Pad2D(
                [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self._padding,
            self._dilation,
            self._groups,
        )
        return x


class Identity(nn.Layer):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return x


def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients."""
    params_dict = {
        # Coefficients:   width,depth,resolution,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder:
    """Block Decoder for readability, straight from the official TensorFlow repository"""

    @staticmethod
    def _decode_block_string(block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert ('s' in options and len(options['s']) == 1) or (
            len(options['s']) == 2 and options['s'][0] == options['s'][1]
        )

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])],
        )

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(
    width_coefficient=None,
    depth_coefficient=None,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    image_size=None,
    num_classes=1000,
):
    """Get block arguments according to parameter and coefficients."""
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model"""
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(
            width_coefficient=w,
            depth_coefficient=d,
            dropout_rate=p,
            image_size=s,
        )
    else:
        raise NotImplementedError(
            'model name is not pre-defined: %s' % model_name
        )
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


url_map = {
    'efficientnet-b0': '/home/aistudio/data/weights/efficientnet-b0-355c32eb.pdparams',
    'efficientnet-b1': '/home/aistudio/data/weights/efficientnet-b1-f1951068.pdparams',
    'efficientnet-b2': '/home/aistudio/data/weights/efficientnet-b2-8bb594d6.pdparams',
    'efficientnet-b3': '/home/aistudio/data/weights/efficientnet-b3-5fb5a3c3.pdparams',
    'efficientnet-b4': '/home/aistudio/data/weights/efficientnet-b4-6ed6700e.pdparams',
    'efficientnet-b5': '/home/aistudio/data/weights/efficientnet-b5-b6417697.pdparams',
    'efficientnet-b6': '/home/aistudio/data/weights/efficientnet-b6-c76e70fd.pdparams',
    'efficientnet-b7': '/home/aistudio/data/weights/efficientnet-b7-dcc49843.pdparams',
}

url_map_advprop = {
    'efficientnet-b0': '/home/aistudio/data/weights/adv-efficientnet-b0-b64d5a18.pdparams',
    'efficientnet-b1': '/home/aistudio/data/weights/adv-efficientnet-b1-0f3ce85a.pdparams',
    'efficientnet-b2': '/home/aistudio/data/weights/adv-efficientnet-b2-6e9d97e5.pdparams',
    'efficientnet-b3': '/home/aistudio/data/weights/adv-efficientnet-b3-cdd7c0f4.pdparams',
    'efficientnet-b4': '/home/aistudio/data/weights/adv-efficientnet-b4-44fb3a87.pdparams',
    'efficientnet-b5': '/home/aistudio/data/weights/adv-efficientnet-b5-86493f6b.pdparams',
    'efficientnet-b6': '/home/aistudio/data/weights/adv-efficientnet-b6-ac80338e.pdparams',
    'efficientnet-b7': '/home/aistudio/data/weights/adv-efficientnet-b7-4652b6dd.pdparams',
    'efficientnet-b8': '/home/aistudio/data/weights/adv-efficientnet-b8-22a8fe65.pdparams',
}


def load_pretrained_weights(
    model, model_name, weights_path=None, load_fc=True, advprop=False
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): The whole model of efficientnet.
        model_name (str): Model name of efficientnet.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        advprop (bool): Whether to load pretrained weights
                        trained with advprop (valid when weights_path is None).
    """

    # AutoAugment or Advprop (different preprocessing)
    url_map_ = url_map_advprop if advprop else url_map
    state_dict = paddle.load(url_map_[model_name])

    if load_fc:
        model.set_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        model.set_state_dict(state_dict)

    print('Loaded pretrained weights for {}'.format(model_name))
