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

import numpy as np
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype
from ..fluid import core, layers
from ..fluid.layers import nn, utils
from ..nn import Layer, Conv2D, Sequential, ReLU, BatchNorm2D
from ..fluid.initializer import Normal
from ..fluid.framework import _non_static_mode, in_dygraph_mode, _in_legacy_dygraph
from paddle.common_ops_import import *
from paddle import _C_ops

__all__ = [ #noqa
    'yolo_loss',
    'yolo_box',
    'deform_conv2d',
    'DeformConv2D',
    'read_file',
    'decode_jpeg',
    'roi_pool',
    'RoIPool',
    'psroi_pool',
    'PSRoIPool',
    'roi_align',
    'RoIAlign',
    'nms',
]


def yolo_loss(x,
              gt_box,
              gt_label,
              anchors,
              anchor_mask,
              class_num,
              ignore_thresh,
              downsample_ratio,
              gt_score=None,
              use_label_smooth=True,
              name=None,
              scale_x_y=1.):
    r"""

    This operator generates YOLOv3 loss based on given predict result and ground
    truth boxes.
    
    The output of previous network is in shape [N, C, H, W], while H and W
    should be the same, H and W specify the grid size, each grid point predict 
    given number bounding boxes, this given number, which following will be represented as S,
    is specified by the number of anchor clusters in each scale. In the second dimension(the channel
    dimension), C should be equal to S * (class_num + 5), class_num is the object 
    category number of source dataset(such as 80 in coco dataset), so in the 
    second(channel) dimension, apart from 4 box location coordinates x, y, w, h, 
    also includes confidence score of the box and class one-hot key of each anchor box.

    Assume the 4 location coordinates are :math:`t_x, t_y, t_w, t_h`, the box predictions
    should be as follows:

    $$
    b_x = \\sigma(t_x) + c_x
    $$
    $$
    b_y = \\sigma(t_y) + c_y
    $$
    $$
    b_w = p_w e^{t_w}
    $$
    $$
    b_h = p_h e^{t_h}
    $$

    In the equation above, :math:`c_x, c_y` is the left top corner of current grid
    and :math:`p_w, p_h` is specified by anchors.

    As for confidence score, it is the logistic regression value of IoU between
    anchor boxes and ground truth boxes, the score of the anchor box which has 
    the max IoU should be 1, and if the anchor box has IoU bigger than ignore 
    thresh, the confidence score loss of this anchor box will be ignored.

    Therefore, the YOLOv3 loss consists of three major parts: box location loss,
    objectness loss and classification loss. The L1 loss is used for 
    box coordinates (w, h), sigmoid cross entropy loss is used for box 
    coordinates (x, y), objectness loss and classification loss.

    Each groud truth box finds a best matching anchor box in all anchors. 
    Prediction of this anchor box will incur all three parts of losses, and
    prediction of anchor boxes with no GT box matched will only incur objectness
    loss.

    In order to trade off box coordinate losses between big boxes and small 
    boxes, box coordinate losses will be mutiplied by scale weight, which is
    calculated as follows.

    $$
    weight_{box} = 2.0 - t_w * t_h
    $$

    Final loss will be represented as follows.

    $$
    loss = (loss_{xy} + loss_{wh}) * weight_{box} + loss_{conf} + loss_{class}
    $$

    While :attr:`use_label_smooth` is set to be :attr:`True`, the classification
    target will be smoothed when calculating classification loss, target of 
    positive samples will be smoothed to :math:`1.0 - 1.0 / class\_num` and target of
    negetive samples will be smoothed to :math:`1.0 / class\_num`.

    While :attr:`gt_score` is given, which means the mixup score of ground truth 
    boxes, all losses incured by a ground truth box will be multiplied by its 
    mixup score.

    Args:
        x (Tensor): The input tensor of YOLOv3 loss operator, This is a 4-D
                      tensor with shape of [N, C, H, W]. H and W should be same,
                      and the second dimension(C) stores box locations, confidence
                      score and classification one-hot keys of each anchor box.
                      The data type is float32 or float64. 
        gt_box (Tensor): groud truth boxes, should be in shape of [N, B, 4],
                          in the third dimension, x, y, w, h should be stored. 
                          x,y is the center coordinate of boxes, w, h are the
                          width and height, x, y, w, h should be divided by 
                          input image height to scale to [0, 1].
                          N is the batch number and B is the max box number in 
                          an image.The data type is float32 or float64. 
        gt_label (Tensor): class id of ground truth boxes, should be in shape
                            of [N, B].The data type is int32. 
        anchors (list|tuple): The anchor width and height, it will be parsed
                              pair by pair.
        anchor_mask (list|tuple): The mask index of anchors used in current
                                  YOLOv3 loss calculation.
        class_num (int): The number of classes.
        ignore_thresh (float): The ignore threshold to ignore confidence loss.
        downsample_ratio (int): The downsample ratio from network input to YOLOv3
                                loss input, so 32, 16, 8 should be set for the
                                first, second, and thrid YOLOv3 loss operators. 
        name (string): The default value is None.  Normally there is no need 
                       for user to set this property.  For more information, 
                       please refer to :ref:`api_guide_Name`
        gt_score (Tensor): mixup score of ground truth boxes, should be in shape
                            of [N, B]. Default None.
        use_label_smooth (bool): Whether to use label smooth. Default True. 
        scale_x_y (float): Scale the center point of decoded bounding box.
                           Default 1.0

    Returns:
        Tensor: A 1-D tensor with shape [N], the value of yolov3 loss

    Raises:
        TypeError: Input x of yolov3_loss must be Tensor
        TypeError: Input gtbox of yolov3_loss must be Tensor 
        TypeError: Input gtlabel of yolov3_loss must be Tensor 
        TypeError: Input gtscore of yolov3_loss must be None or Tensor 
        TypeError: Attr anchors of yolov3_loss must be list or tuple
        TypeError: Attr class_num of yolov3_loss must be an integer
        TypeError: Attr ignore_thresh of yolov3_loss must be a float number
        TypeError: Attr use_label_smooth of yolov3_loss must be a bool value

    Examples:
      .. code-block:: python

          import paddle
          import numpy as np

          x = np.random.random([2, 14, 8, 8]).astype('float32')
          gt_box = np.random.random([2, 10, 4]).astype('float32')
          gt_label = np.random.random([2, 10]).astype('int32')

          x = paddle.to_tensor(x)
          gt_box = paddle.to_tensor(gt_box)
          gt_label = paddle.to_tensor(gt_label)

          loss = paddle.vision.ops.yolo_loss(x,
                                             gt_box=gt_box,
                                             gt_label=gt_label,
                                             anchors=[10, 13, 16, 30],
                                             anchor_mask=[0, 1],
                                             class_num=2,
                                             ignore_thresh=0.7,
                                             downsample_ratio=8,
                                             use_label_smooth=True,
                                             scale_x_y=1.)
    """

    if _non_static_mode():
        loss, _, _ = _C_ops.yolov3_loss(
            x, gt_box, gt_label, gt_score, 'anchors', anchors, 'anchor_mask',
            anchor_mask, 'class_num', class_num, 'ignore_thresh', ignore_thresh,
            'downsample_ratio', downsample_ratio, 'use_label_smooth',
            use_label_smooth, 'scale_x_y', scale_x_y)
        return loss

    helper = LayerHelper('yolov3_loss', **locals())

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'yolo_loss')
    check_variable_and_dtype(gt_box, 'gt_box', ['float32', 'float64'],
                             'yolo_loss')
    check_variable_and_dtype(gt_label, 'gt_label', 'int32', 'yolo_loss')
    check_type(anchors, 'anchors', (list, tuple), 'yolo_loss')
    check_type(anchor_mask, 'anchor_mask', (list, tuple), 'yolo_loss')
    check_type(class_num, 'class_num', int, 'yolo_loss')
    check_type(ignore_thresh, 'ignore_thresh', float, 'yolo_loss')
    check_type(use_label_smooth, 'use_label_smooth', bool, 'yolo_loss')

    loss = helper.create_variable_for_type_inference(dtype=x.dtype)

    objectness_mask = helper.create_variable_for_type_inference(dtype='int32')
    gt_match_mask = helper.create_variable_for_type_inference(dtype='int32')

    inputs = {
        "X": x,
        "GTBox": gt_box,
        "GTLabel": gt_label,
    }
    if gt_score is not None:
        inputs["GTScore"] = gt_score

    attrs = {
        "anchors": anchors,
        "anchor_mask": anchor_mask,
        "class_num": class_num,
        "ignore_thresh": ignore_thresh,
        "downsample_ratio": downsample_ratio,
        "use_label_smooth": use_label_smooth,
        "scale_x_y": scale_x_y,
    }

    helper.append_op(
        type='yolov3_loss',
        inputs=inputs,
        outputs={
            'Loss': loss,
            'ObjectnessMask': objectness_mask,
            'GTMatchMask': gt_match_mask
        },
        attrs=attrs)
    return loss


def yolo_box(x,
             img_size,
             anchors,
             class_num,
             conf_thresh,
             downsample_ratio,
             clip_bbox=True,
             name=None,
             scale_x_y=1.,
             iou_aware=False,
             iou_aware_factor=0.5):
    r"""

    This operator generates YOLO detection boxes from output of YOLOv3 network.
    
    The output of previous network is in shape [N, C, H, W], while H and W
    should be the same, H and W specify the grid size, each grid point predict 
    given number boxes, this given number, which following will be represented as S,
    is specified by the number of anchors. In the second dimension(the channel
    dimension), C should be equal to S * (5 + class_num) if :attr:`iou_aware` is false,
    otherwise C should be equal to S * (6 + class_num). class_num is the object
    category number of source dataset(such as 80 in coco dataset), so the 
    second(channel) dimension, apart from 4 box location coordinates x, y, w, h, 
    also includes confidence score of the box and class one-hot key of each anchor 
    box.

    Assume the 4 location coordinates are :math:`t_x, t_y, t_w, t_h`, the box 
    predictions should be as follows:

    $$
    b_x = \\sigma(t_x) + c_x
    $$
    $$
    b_y = \\sigma(t_y) + c_y
    $$
    $$
    b_w = p_w e^{t_w}
    $$
    $$
    b_h = p_h e^{t_h}
    $$

    in the equation above, :math:`c_x, c_y` is the left top corner of current grid
    and :math:`p_w, p_h` is specified by anchors.

    The logistic regression value of the 5th channel of each anchor prediction boxes
    represents the confidence score of each prediction box, and the logistic
    regression value of the last :attr:`class_num` channels of each anchor prediction 
    boxes represents the classifcation scores. Boxes with confidence scores less than
    :attr:`conf_thresh` should be ignored, and box final scores is the product of 
    confidence scores and classification scores.

    $$
    score_{pred} = score_{conf} * score_{class}
    $$

    where the confidence scores follow the formula bellow

    .. math::

        score_{conf} = \begin{case}
                         obj, \text{if } iou_aware == flase \\
                         obj^{1 - iou_aware_factor} * iou^{iou_aware_factor}, \text{otherwise}
                       \end{case}

    Args:
        x (Tensor): The input tensor of YoloBox operator is a 4-D tensor with
                      shape of [N, C, H, W]. The second dimension(C) stores box
                      locations, confidence score and classification one-hot keys
                      of each anchor box. Generally, X should be the output of
                      YOLOv3 network. The data type is float32 or float64. 
        img_size (Tensor): The image size tensor of YoloBox operator, This is a
                           2-D tensor with shape of [N, 2]. This tensor holds
                           height and width of each input image used for resizing
                           output box in input image scale. The data type is int32. 
        anchors (list|tuple): The anchor width and height, it will be parsed pair
                              by pair.
        class_num (int): The number of classes.
        conf_thresh (float): The confidence scores threshold of detection boxes.
                             Boxes with confidence scores under threshold should
                             be ignored.
        downsample_ratio (int): The downsample ratio from network input to
                                :attr:`yolo_box` operator input, so 32, 16, 8
                                should be set for the first, second, and thrid
                                :attr:`yolo_box` layer.
        clip_bbox (bool): Whether clip output bonding box in :attr:`img_size`
                          boundary. Default true.
        scale_x_y (float): Scale the center point of decoded bounding box.
                           Default 1.0
        name (string): The default value is None.  Normally there is no need 
                       for user to set this property.  For more information, 
                       please refer to :ref:`api_guide_Name`
        iou_aware (bool): Whether use iou aware. Default false
        iou_aware_factor (float): iou aware factor. Default 0.5

    Returns:
        Tensor: A 3-D tensor with shape [N, M, 4], the coordinates of boxes,
        and a 3-D tensor with shape [N, M, :attr:`class_num`], the classification 
        scores of boxes.

    Raises:
        TypeError: Input x of yolov_box must be Tensor
        TypeError: Attr anchors of yolo box must be list or tuple
        TypeError: Attr class_num of yolo box must be an integer
        TypeError: Attr conf_thresh of yolo box must be a float number

    Examples:

    .. code-block:: python

        import paddle
        import numpy as np

        x = np.random.random([2, 14, 8, 8]).astype('float32')
        img_size = np.ones((2, 2)).astype('int32')

        x = paddle.to_tensor(x)
        img_size = paddle.to_tensor(img_size)

        boxes, scores = paddle.vision.ops.yolo_box(x,
                                                   img_size=img_size,
                                                   anchors=[10, 13, 16, 30],
                                                   class_num=2,
                                                   conf_thresh=0.01,
                                                   downsample_ratio=8,
                                                   clip_bbox=True,
                                                   scale_x_y=1.)
    """
    if in_dygraph_mode():
        boxes, scores = _C_ops.final_state_yolo_box(
            x, img_size, anchors, class_num, conf_thresh, downsample_ratio,
            clip_bbox, scale_x_y, iou_aware, iou_aware_factor)
        return boxes, scores

    if _non_static_mode():
        boxes, scores = _C_ops.yolo_box(
            x, img_size, 'anchors', anchors, 'class_num', class_num,
            'conf_thresh', conf_thresh, 'downsample_ratio', downsample_ratio,
            'clip_bbox', clip_bbox, 'scale_x_y', scale_x_y, 'iou_aware',
            iou_aware, 'iou_aware_factor', iou_aware_factor)
        return boxes, scores

    helper = LayerHelper('yolo_box', **locals())

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'yolo_box')
    check_variable_and_dtype(img_size, 'img_size', 'int32', 'yolo_box')
    check_type(anchors, 'anchors', (list, tuple), 'yolo_box')
    check_type(conf_thresh, 'conf_thresh', float, 'yolo_box')

    boxes = helper.create_variable_for_type_inference(dtype=x.dtype)
    scores = helper.create_variable_for_type_inference(dtype=x.dtype)

    attrs = {
        "anchors": anchors,
        "class_num": class_num,
        "conf_thresh": conf_thresh,
        "downsample_ratio": downsample_ratio,
        "clip_bbox": clip_bbox,
        "scale_x_y": scale_x_y,
        "iou_aware": iou_aware,
        "iou_aware_factor": iou_aware_factor
    }

    helper.append_op(
        type='yolo_box',
        inputs={
            "X": x,
            "ImgSize": img_size,
        },
        outputs={
            'Boxes': boxes,
            'Scores': scores,
        },
        attrs=attrs)
    return boxes, scores


def deform_conv2d(x,
                  offset,
                  weight,
                  bias=None,
                  stride=1,
                  padding=0,
                  dilation=1,
                  deformable_groups=1,
                  groups=1,
                  mask=None,
                  name=None):
    r"""
    Compute 2-D deformable convolution on 4-D input.
    Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:


    Deformable Convolution v2:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}

    Deformable Convolution v1:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}

    Where :math:`\Delta p_k` and :math:`\Delta m_k` are the learnable offset and modulation scalar for the k-th location,
    Which :math:`\Delta m_k` is one in deformable convolution v1. Please refer to `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168v2>`_ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.

    Example:
        - Input:

          x shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          weight shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

          offset shape: :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`

          mask shape: :math:`(N, H_f * W_f, H_{out}, W_{out})`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        x (Tensor): The input image with [N, C, H, W] format. A Tensor with type
            float32, float64.
        offset (Tensor): The input coordinate offset of deformable convolution layer.
            A Tensor with type float32, float64.
        weight (Tensor): The convolution kernel with shape [M, C/g, kH, kW], where M is
            the number of output channels, g is the number of groups, kH is the filter's
            height, kW is the filter's width.
        bias (Tensor, optional): The bias with shape [M,].
        stride (int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        padding (int|list|tuple, optional): The padding size. If padding is a list/tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        deformable_groups (int): The number of deformable group partitions.
            Default: deformable_groups = 1.
        groups (int, optonal): The groups number of the deformable conv layer. According to
            grouped convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        mask (Tensor, optional): The input mask of deformable convolution layer.
            A Tensor with type float32, float64. It should be None when you use
            deformable convolution v1.
        name(str, optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.
    Returns:
        Tensor: The tensor variable storing the deformable convolution \
                  result. A Tensor with type float32, float64.
    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.
    Examples:
        .. code-block:: python

          #deformable conv v2:

          import paddle
          input = paddle.rand((8, 1, 28, 28))
          kh, kw = 3, 3
          weight = paddle.rand((16, 1, kh, kw))
          # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
          # mask shape should be [bs, hw * hw, out_h, out_w]
          # In this case, for an input of 28, stride of 1
          # and kernel size of 3, without padding, the output size is 26
          offset = paddle.rand((8, 2 * kh * kw, 26, 26))
          mask = paddle.rand((8, kh * kw, 26, 26))
          out = paddle.vision.ops.deform_conv2d(input, offset, weight, mask=mask)
          print(out.shape)
          # returns
          [8, 16, 26, 26]

          #deformable conv v1:

          import paddle
          input = paddle.rand((8, 1, 28, 28))
          kh, kw = 3, 3
          weight = paddle.rand((16, 1, kh, kw))
          # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
          # In this case, for an input of 28, stride of 1
          # and kernel size of 3, without padding, the output size is 26
          offset = paddle.rand((8, 2 * kh * kw, 26, 26))
          out = paddle.vision.ops.deform_conv2d(input, offset, weight)
          print(out.shape)
          # returns
          [8, 16, 26, 26]
    """
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    use_deform_conv2d_v1 = True if mask is None else False

    if in_dygraph_mode():
        pre_bias = _C_ops.final_state_deformable_conv(
            x, offset, weight, mask, stride, padding, dilation,
            deformable_groups, groups, 1)
        if bias is not None:
            out = nn.elementwise_add(pre_bias, bias, axis=1)
        else:
            out = pre_bias
    elif _in_legacy_dygraph():
        attrs = ('strides', stride, 'paddings', padding, 'dilations', dilation,
                 'deformable_groups', deformable_groups, 'groups', groups,
                 'im2col_step', 1)
        if use_deform_conv2d_v1:
            op_type = 'deformable_conv_v1'
            pre_bias = getattr(_C_ops, op_type)(x, offset, weight, *attrs)
        else:
            op_type = 'deformable_conv'
            pre_bias = getattr(_C_ops, op_type)(x, offset, mask, weight, *attrs)
        if bias is not None:
            out = nn.elementwise_add(pre_bias, bias, axis=1)
        else:
            out = pre_bias
    else:
        check_variable_and_dtype(x, "x", ['float32', 'float64'],
                                 'deform_conv2d')
        check_variable_and_dtype(offset, "offset", ['float32', 'float64'],
                                 'deform_conv2d')

        num_channels = x.shape[1]

        helper = LayerHelper('deformable_conv', **locals())
        dtype = helper.input_dtype()

        stride = utils.convert_to_list(stride, 2, 'stride')
        padding = utils.convert_to_list(padding, 2, 'padding')
        dilation = utils.convert_to_list(dilation, 2, 'dilation')

        pre_bias = helper.create_variable_for_type_inference(dtype)

        if use_deform_conv2d_v1:
            op_type = 'deformable_conv_v1'
            inputs = {
                'Input': x,
                'Filter': weight,
                'Offset': offset,
            }
        else:
            op_type = 'deformable_conv'
            inputs = {
                'Input': x,
                'Filter': weight,
                'Offset': offset,
                'Mask': mask,
            }

        outputs = {"Output": pre_bias}
        attrs = {
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'deformable_groups': deformable_groups,
            'im2col_step': 1,
        }
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)

        if bias is not None:
            out = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [bias]},
                outputs={'Out': [out]},
                attrs={'axis': 1})
        else:
            out = pre_bias
    return out


class DeformConv2D(Layer):
    r"""
    Compute 2-D deformable convolution on 4-D input.
    Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:


    Deformable Convolution v2:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}

    Deformable Convolution v1:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}

    Where :math:`\Delta p_k` and :math:`\Delta m_k` are the learnable offset and modulation scalar for the k-th location,
    Which :math:`\Delta m_k` is one in deformable convolution v1. Please refer to `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168v2>`_ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.

    Example:
        - Input:

          x shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          weight shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

          offset shape: :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`

          mask shape: :math:`(N, H_f * W_f, H_{out}, W_{out})`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\
            W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1


    Parameters:
        in_channels(int): The number of input channels in the input image.
        out_channels(int): The number of output channels produced by the convolution.
        kernel_size(int|list|tuple): The size of the convolving kernel.
        stride(int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain three integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. The default value is 1.
        padding (int|list|tuple, optional): The padding size. If padding is a list/tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation(int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. The default value is 1.
        deformable_groups (int): The number of deformable group partitions.
            Default: deformable_groups = 1.
        groups(int, optional): The groups number of the Conv3D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. The default value is 1.
        weight_attr(ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`. The default value is None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
    Attribute:
        **weight** (Parameter): the learnable weights of filter of this layer.
        **bias** (Parameter or None): the learnable bias of this layer.
    Shape:
        - x: :math:`(N, C_{in}, H_{in}, W_{in})`
        - offset: :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`
        - mask: :math:`(N, H_f * W_f, H_{out}, W_{out})`
        - output: :math:`(N, C_{out}, H_{out}, W_{out})`
        
        Where
        
        ..  math::

            H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1 \\
            W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

    Examples:
        .. code-block:: python

          #deformable conv v2:

          import paddle
          input = paddle.rand((8, 1, 28, 28))
          kh, kw = 3, 3
          # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
          # mask shape should be [bs, hw * hw, out_h, out_w]
          # In this case, for an input of 28, stride of 1
          # and kernel size of 3, without padding, the output size is 26
          offset = paddle.rand((8, 2 * kh * kw, 26, 26))
          mask = paddle.rand((8, kh * kw, 26, 26))
          deform_conv = paddle.vision.ops.DeformConv2D(
              in_channels=1,
              out_channels=16,
              kernel_size=[kh, kw])
          out = deform_conv(input, offset, mask)
          print(out.shape)
          # returns
          [8, 16, 26, 26]

          #deformable conv v1:

          import paddle
          input = paddle.rand((8, 1, 28, 28))
          kh, kw = 3, 3
          # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
          # mask shape should be [bs, hw * hw, out_h, out_w]
          # In this case, for an input of 28, stride of 1
          # and kernel size of 3, without padding, the output size is 26
          offset = paddle.rand((8, 2 * kh * kw, 26, 26))
          deform_conv = paddle.vision.ops.DeformConv2D(
              in_channels=1,
              out_channels=16,
              kernel_size=[kh, kw])
          out = deform_conv(input, offset)
          print(out.shape)
          # returns
          [8, 16, 26, 26]
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 deformable_groups=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformConv2D, self).__init__()
        assert weight_attr is not False, "weight_attr should not be False in Conv."
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._deformable_groups = deformable_groups
        self._groups = groups
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._channel_dim = 1

        self._stride = utils.convert_to_list(stride, 2, 'stride')
        self._dilation = utils.convert_to_list(dilation, 2, 'dilation')
        self._kernel_size = utils.convert_to_list(kernel_size, 2, 'kernel_size')

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")

        self._padding = utils.convert_to_list(padding, 2, 'padding')

        filter_shape = [out_channels, in_channels // groups] + self._kernel_size

        def _get_default_param_initializer():
            filter_elem_num = np.prod(self._kernel_size) * self._in_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std, 0)

        self.weight = self.create_parameter(
            shape=filter_shape,
            attr=self._weight_attr,
            default_initializer=_get_default_param_initializer())
        self.bias = self.create_parameter(
            attr=self._bias_attr, shape=[self._out_channels], is_bias=True)

    def forward(self, x, offset, mask=None):
        out = deform_conv2d(
            x=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            deformable_groups=self._deformable_groups,
            groups=self._groups,
            mask=mask)
        return out


def read_file(filename, name=None):
    """
    Reads and outputs the bytes contents of a file as a uint8 Tensor
    with one dimension.

    Args:
        filename (str): Path of the file to be read.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        A uint8 tensor.

    Examples:
        .. code-block:: python

            import cv2
            import paddle

            fake_img = (paddle.rand((400, 300, 3)).numpy() * 255).astype('uint8')
            
            cv2.imwrite('fake.jpg', fake_img)

            img_bytes = paddle.vision.ops.read_file('fake.jpg')
            
            print(img_bytes.shape)
            # [142915]
    """

    if _non_static_mode():
        return _C_ops.read_file('filename', filename)

    inputs = dict()
    attrs = {'filename': filename}

    helper = LayerHelper("read_file", **locals())
    out = helper.create_variable_for_type_inference('uint8')
    helper.append_op(
        type="read_file", inputs=inputs, attrs=attrs, outputs={"Out": out})

    return out


def decode_jpeg(x, mode='unchanged', name=None):
    """
    Decodes a JPEG image into a 3 dimensional RGB Tensor or 1 dimensional Gray Tensor. 
    Optionally converts the image to the desired format. 
    The values of the output tensor are uint8 between 0 and 255.

    Args:
        x (Tensor): A one dimensional uint8 tensor containing the raw bytes 
            of the JPEG image.
        mode (str): The read mode used for optionally converting the image. 
            Default: 'unchanged'.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: A decoded image tensor with shape (imge_channels, image_height, image_width)

    Examples:
        .. code-block:: python

            # required: gpu
            import cv2
            import numpy as np
            import paddle

            fake_img = (np.random.random(
                        (400, 300, 3)) * 255).astype('uint8')

            cv2.imwrite('fake.jpg', fake_img)

            img_bytes = paddle.vision.ops.read_file('fake.jpg')
            img = paddle.vision.ops.decode_jpeg(img_bytes)

            print(img.shape)
    """

    if _non_static_mode():
        return _C_ops.decode_jpeg(x, "mode", mode)

    inputs = {'X': x}
    attrs = {"mode": mode}

    helper = LayerHelper("decode_jpeg", **locals())
    out = helper.create_variable_for_type_inference('uint8')
    helper.append_op(
        type="decode_jpeg", inputs=inputs, attrs=attrs, outputs={"Out": out})

    return out


def psroi_pool(x, boxes, boxes_num, output_size, spatial_scale=1.0, name=None):
    """
    Position sensitive region of interest pooling (also known as PSROIPooling) is to perform
    position-sensitive average pooling on regions of interest specified by input. It performs 
    on inputs of nonuniform sizes to obtain fixed-size feature maps.

    PSROIPooling is proposed by R-FCN. Please refer to https://arxiv.org/abs/1605.06409 for more details.

    Args:
        x (Tensor): Input features with shape (N, C, H, W). The data type can be float32 or float64.
        boxes (Tensor): Box coordinates of ROIs (Regions of Interest) to pool over. It should be
                         a 2-D Tensor with shape (num_rois, 4). Given as [[x1, y1, x2, y2], ...], 
                         (x1, y1) is the top left coordinates, and (x2, y2) is the bottom
                         right coordinates.
        boxes_num (Tensor): The number of boxes contained in each picture in the batch.
        output_size (int|Tuple(int, int))  The pooled output size(H, W), data type 
                               is int32. If int, H and W are both equal to output_size.
        spatial_scale (float): Multiplicative spatial scale factor to translate ROI coords from their 
                               input scale to the scale used when pooling. Default: 1.0
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        4-D Tensor. The pooled ROIs with shape (num_rois, output_channels, pooled_h, pooled_w).
        The output_channels equal to C / (pooled_h * pooled_w), where C is the channels of input.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.uniform([2, 490, 28, 28], dtype='float32')
            boxes = paddle.to_tensor([[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]], dtype='float32')
            boxes_num = paddle.to_tensor([1, 2], dtype='int32')
            pool_out = paddle.vision.ops.psroi_pool(x, boxes, boxes_num, 7, 1.0)
    """

    check_type(output_size, 'output_size', (int, tuple, list), 'psroi_pool')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    pooled_height, pooled_width = output_size
    assert len(x.shape) == 4, \
            "Input features with shape should be (N, C, H, W)"
    output_channels = int(x.shape[1] / (pooled_height * pooled_width))
    if in_dygraph_mode():
        return _C_ops.final_state_psroi_pool(x, boxes, boxes_num, pooled_height,
                                             pooled_width, output_channels,
                                             spatial_scale)
    if _in_legacy_dygraph():
        return _C_ops.psroi_pool(x, boxes, boxes_num, "output_channels",
                                 output_channels, "spatial_scale",
                                 spatial_scale, "pooled_height", pooled_height,
                                 "pooled_width", pooled_width)

    helper = LayerHelper('psroi_pool', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='psroi_pool',
        inputs={'X': x,
                'ROIs': boxes},
        outputs={'Out': out},
        attrs={
            'output_channels': output_channels,
            'spatial_scale': spatial_scale,
            'pooled_height': pooled_height,
            'pooled_width': pooled_width
        })
    return out


class PSRoIPool(Layer):
    """
    This interface is used to construct a callable object of the ``PSRoIPool`` class. Please
    refer to :ref:`api_paddle_vision_ops_psroi_pool`.

    Args:
        output_size (int|Tuple(int, int))  The pooled output size(H, W), data type 
                               is int32. If int, H and W are both equal to output_size.
        spatial_scale (float): Multiplicative spatial scale factor to translate ROI coords from their 
                               input scale to the scale used when pooling. Default: 1.0.

    Shape:
        - x: 4-D Tensor with shape (N, C, H, W).
        - boxes: 2-D Tensor with shape (num_rois, 4).
        - boxes_num: 1-D Tensor.
        - output: 4-D tensor with shape (num_rois, output_channels, pooled_h, pooled_w).
              The output_channels equal to C / (pooled_h * pooled_w), where C is the channels of input.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle
            
            psroi_module = paddle.vision.ops.PSRoIPool(7, 1.0)
            x = paddle.uniform([2, 490, 28, 28], dtype='float32')
            boxes = paddle.to_tensor([[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]], dtype='float32')
            boxes_num = paddle.to_tensor([1, 2], dtype='int32')
            pool_out = psroi_module(x, boxes, boxes_num)

    """

    def __init__(self, output_size, spatial_scale=1.0):
        super(PSRoIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, x, boxes, boxes_num):
        return psroi_pool(x, boxes, boxes_num, self.output_size,
                          self.spatial_scale)


def roi_pool(x, boxes, boxes_num, output_size, spatial_scale=1.0, name=None):
    """
    This operator implements the roi_pooling layer.
    Region of interest pooling (also known as RoI pooling) is to perform max pooling on inputs of nonuniform sizes to obtain fixed-size feature maps (e.g. 7*7).
    The operator has three steps: 1. Dividing each region proposal into equal-sized sections with output_size(h, w) 2. Finding the largest value in each section 3. Copying these max values to the output buffer  
    For more information, please refer to https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn.

    Args:
        x (Tensor): input feature, 4D-Tensor with the shape of [N,C,H,W], 
            where N is the batch size, C is the input channel, H is Height, W is weight. 
            The data type is float32 or float64.
        boxes (Tensor): boxes (Regions of Interest) to pool over. 
            2D-Tensor with the shape of [num_boxes,4]. 
            Given as [[x1, y1, x2, y2], ...], (x1, y1) is the top left coordinates, 
            and (x2, y2) is the bottom right coordinates.
        boxes_num (Tensor): the number of RoIs in each image, data type is int32. Default: None
        output_size (int or tuple[int, int]): the pooled output size(h, w), data type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float, optional): multiplicative spatial scale factor to translate ROI coords from their input scale to the scale used when pooling. Default: 1.0
        name(str, optional): for detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and None by default.

    Returns:
        pool_out (Tensor): the pooled feature, 4D-Tensor with the shape of [num_boxes, C, output_size[0], output_size[1]].  

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.ops import roi_pool

            data = paddle.rand([1, 256, 32, 32])
            boxes = paddle.rand([3, 4])
            boxes[:, 2] += boxes[:, 0] + 3
            boxes[:, 3] += boxes[:, 1] + 4
            boxes_num = paddle.to_tensor([3]).astype('int32')
            pool_out = roi_pool(data, boxes, boxes_num=boxes_num, output_size=3)
            assert pool_out.shape == [3, 256, 3, 3], ''
    """

    check_type(output_size, 'output_size', (int, tuple), 'roi_pool')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size
    if in_dygraph_mode():
        assert boxes_num is not None, "boxes_num should not be None in dygraph mode."
        return _C_ops.final_state_roi_pool(x, boxes, boxes_num, pooled_height,
                                           pooled_width, spatial_scale)
    if _in_legacy_dygraph():
        assert boxes_num is not None, "boxes_num should not be None in dygraph mode."
        pool_out, argmaxes = _C_ops.roi_pool(
            x, boxes, boxes_num, "pooled_height", pooled_height, "pooled_width",
            pooled_width, "spatial_scale", spatial_scale)
        return pool_out

    else:
        check_variable_and_dtype(x, 'x', ['float32'], 'roi_pool')
        check_variable_and_dtype(boxes, 'boxes', ['float32'], 'roi_pool')
        helper = LayerHelper('roi_pool', **locals())
        dtype = helper.input_dtype()
        pool_out = helper.create_variable_for_type_inference(dtype)
        argmaxes = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {
            "X": x,
            "ROIs": boxes,
        }
        if boxes_num is not None:
            inputs['RoisNum'] = boxes_num
        helper.append_op(
            type="roi_pool",
            inputs=inputs,
            outputs={"Out": pool_out,
                     "Argmax": argmaxes},
            attrs={
                "pooled_height": pooled_height,
                "pooled_width": pooled_width,
                "spatial_scale": spatial_scale
            })
        return pool_out


class RoIPool(Layer):
    """
    This interface is used to construct a callable object of the `RoIPool` class. Please
    refer to :ref:`api_paddle_vision_ops_roi_pool`.  

    Args:
        output_size (int or tuple[int, int]): the pooled output size(h, w), data type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float, optional): multiplicative spatial scale factor to translate ROI coords from their input scale to the scale used when pooling. Default: 1.0.

    Returns:
        pool_out (Tensor): the pooled feature, 4D-Tensor with the shape of [num_boxes, C, output_size[0], output_size[1]].  

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.ops import RoIPool
            
            data = paddle.rand([1, 256, 32, 32])
            boxes = paddle.rand([3, 4])
            boxes[:, 2] += boxes[:, 0] + 3
            boxes[:, 3] += boxes[:, 1] + 4
            boxes_num = paddle.to_tensor([3]).astype('int32')
            roi_pool = RoIPool(output_size=(4, 3))
            pool_out = roi_pool(data, boxes, boxes_num)
            assert pool_out.shape == [3, 256, 4, 3], ''
    """

    def __init__(self, output_size, spatial_scale=1.0):
        super(RoIPool, self).__init__()
        self._output_size = output_size
        self._spatial_scale = spatial_scale

    def forward(self, x, boxes, boxes_num):
        return roi_pool(
            x=x,
            boxes=boxes,
            boxes_num=boxes_num,
            output_size=self._output_size,
            spatial_scale=self._spatial_scale)

    def extra_repr(self):
        main_str = 'output_size={_output_size}, spatial_scale={_spatial_scale}'
        return main_str.format(**self.__dict__)


def roi_align(x,
              boxes,
              boxes_num,
              output_size,
              spatial_scale=1.0,
              sampling_ratio=-1,
              aligned=True,
              name=None):
    """
    This operator implements the roi_align layer.
    Region of Interest (RoI) Align operator (also known as RoI Align) is to
    perform bilinear interpolation on inputs of nonuniform sizes to obtain
    fixed-size feature maps (e.g. 7*7), as described in Mask R-CNN.

    Dividing each region proposal into equal-sized sections with the pooled_width
    and pooled_height. Location remains the origin result.

    In each ROI bin, the value of the four regularly sampled locations are
    computed directly through bilinear interpolation. The output is the mean of
    four locations. Thus avoid the misaligned problem. 

    Args:
        x (Tensor): Input feature, 4D-Tensor with the shape of [N,C,H,W], 
            where N is the batch size, C is the input channel, H is Height,
            W is weight. The data type is float32 or float64.
        boxes (Tensor): Boxes (RoIs, Regions of Interest) to pool over. It 
            should be a 2-D Tensor of shape (num_boxes, 4). The data type is
            float32 or float64. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
            the top left coordinates, and (x2, y2) is the bottom right coordinates.
        boxes_num (Tensor): The number of boxes contained in each picture in
            the batch, the data type is int32.
        output_size (int or Tuple[int, int]): The pooled output size(h, w), data
            type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float32): Multiplicative spatial scale factor to translate
            ROI coords from their input scale to the scale used when pooling.
            Default: 1.0
        sampling_ratio (int32): number of sampling points in the interpolation
            grid used to compute the output value of each pooled output bin.
            If > 0, then exactly ``sampling_ratio x sampling_ratio`` sampling
            points per bin are used.
            If <= 0, then an adaptive number of grid points are used (computed
            as ``ceil(roi_width / output_width)``, and likewise for height).
            Default: -1
        aligned (bool): If False, use the legacy implementation. If True, pixel
            shift the box coordinates it by -0.5 for a better alignment with the
            two neighboring pixel indices. This version is used in Detectron2.
            Default: True
        name(str, optional): For detailed information, please refer to :
            ref:`api_guide_Name`. Usually name is no need to set and None by
            default.

    Returns:
        Tensor: The output of ROIAlignOp is a 4-D tensor with shape (num_boxes,
            channels, pooled_h, pooled_w). The data type is float32 or float64.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.ops import roi_align

            data = paddle.rand([1, 256, 32, 32])
            boxes = paddle.rand([3, 4])
            boxes[:, 2] += boxes[:, 0] + 3
            boxes[:, 3] += boxes[:, 1] + 4
            boxes_num = paddle.to_tensor([3]).astype('int32')
            align_out = roi_align(data, boxes, boxes_num, output_size=3)
            assert align_out.shape == [3, 256, 3, 3]
    """

    check_type(output_size, 'output_size', (int, tuple), 'roi_align')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size
    if in_dygraph_mode():
        assert boxes_num is not None, "boxes_num should not be None in dygraph mode."
        return _C_ops.final_state_roi_align(x, boxes, boxes_num, pooled_height,
                                            pooled_width, spatial_scale,
                                            sampling_ratio, aligned)
    if _in_legacy_dygraph():
        assert boxes_num is not None, "boxes_num should not be None in dygraph mode."
        align_out = _C_ops.roi_align(
            x, boxes, boxes_num, "pooled_height", pooled_height, "pooled_width",
            pooled_width, "spatial_scale", spatial_scale, "sampling_ratio",
            sampling_ratio, "aligned", aligned)
        return align_out

    else:
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'roi_align')
        check_variable_and_dtype(boxes, 'boxes', ['float32', 'float64'],
                                 'roi_align')
        helper = LayerHelper('roi_align', **locals())
        dtype = helper.input_dtype()
        align_out = helper.create_variable_for_type_inference(dtype)
        inputs = {
            "X": x,
            "ROIs": boxes,
        }
        if boxes_num is not None:
            inputs['RoisNum'] = boxes_num
        helper.append_op(
            type="roi_align",
            inputs=inputs,
            outputs={"Out": align_out},
            attrs={
                "pooled_height": pooled_height,
                "pooled_width": pooled_width,
                "spatial_scale": spatial_scale,
                "sampling_ratio": sampling_ratio,
                "aligned": aligned,
            })
        return align_out


class RoIAlign(Layer):
    """
    This interface is used to construct a callable object of the `RoIAlign` class.
    Please refer to :ref:`api_paddle_vision_ops_roi_align`.

    Args:
        output_size (int or tuple[int, int]): The pooled output size(h, w),
            data type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float32, optional): Multiplicative spatial scale factor
            to translate ROI coords from their input scale to the scale used
            when pooling. Default: 1.0

    Returns:
        align_out (Tensor): The output of ROIAlign operator is a 4-D tensor with
            shape (num_boxes, channels, pooled_h, pooled_w).

    Examples:
        ..  code-block:: python

            import paddle
            from paddle.vision.ops import RoIAlign

            data = paddle.rand([1, 256, 32, 32])
            boxes = paddle.rand([3, 4])
            boxes[:, 2] += boxes[:, 0] + 3
            boxes[:, 3] += boxes[:, 1] + 4
            boxes_num = paddle.to_tensor([3]).astype('int32')
            roi_align = RoIAlign(output_size=(4, 3))
            align_out = roi_align(data, boxes, boxes_num)
            assert align_out.shape == [3, 256, 4, 3]
    """

    def __init__(self, output_size, spatial_scale=1.0):
        super(RoIAlign, self).__init__()
        self._output_size = output_size
        self._spatial_scale = spatial_scale

    def forward(self, x, boxes, boxes_num, aligned=True):
        return roi_align(
            x=x,
            boxes=boxes,
            boxes_num=boxes_num,
            output_size=self._output_size,
            spatial_scale=self._spatial_scale,
            aligned=aligned)


class ConvNormActivation(Sequential):
    """
    Configurable block used for Convolution-Normalzation-Activation blocks.
    This code is based on the torchvision code with modifications.
    You can also see at https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L68
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalzation-Activation block
        kernel_size: (int|list|tuple, optional): Size of the convolving kernel. Default: 3
        stride (int|list|tuple, optional): Stride of the convolution. Default: 1
        padding (int|str|tuple|list, optional): Padding added to all four sides of the input. Default: None,
            in wich case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., paddle.nn.Layer], optional): Norm layer that will be stacked on top of the convolutiuon layer.
            If ``None`` this layer wont be used. Default: ``paddle.nn.BatchNorm2D``
        activation_layer (Callable[..., paddle.nn.Layer], optional): Activation function which will be stacked on top of the normalization
            layer (if not ``None``), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``paddle.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 groups=1,
                 norm_layer=BatchNorm2D,
                 activation_layer=ReLU,
                 dilation=1,
                 bias=None):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias_attr=bias)
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


def nms(boxes,
        iou_threshold=0.3,
        scores=None,
        category_idxs=None,
        categories=None,
        top_k=None):
    r"""
    This operator implements non-maximum suppression. Non-maximum suppression (NMS)
    is used to select one bounding box out of many overlapping bounding boxes in object detection. 
    Boxes with IoU > iou_threshold will be considered as overlapping boxes, 
    just one with highest score can be kept. Here IoU is Intersection Over Union, 
    which can be computed by:

    ..  math::

        IoU = \frac{intersection\_area(box1, box2)}{union\_area(box1, box2)}

    If scores are provided, input boxes will be sorted by their scores firstly.

    If category_idxs and categories are provided, NMS will be performed with a batched style, 
    which means NMS will be applied to each category respectively and results of each category
    will be concated and sorted by scores.
    
    If K is provided, only the first k elements will be returned. Otherwise, all box indices sorted by scores will be returned.

    Args:
        boxes(Tensor): The input boxes data to be computed, it's a 2D-Tensor with 
            the shape of [num_boxes, 4]. The data type is float32 or float64. 
            Given as [[x1, y1, x2, y2], …],  (x1, y1) is the top left coordinates, 
            and (x2, y2) is the bottom right coordinates. 
            Their relation should be ``0 <= x1 < x2 && 0 <= y1 < y2``.
        iou_threshold(float32, optional): IoU threshold for determine overlapping boxes. Default value: 0.3.
        scores(Tensor, optional): Scores corresponding to boxes, it's a 1D-Tensor with 
            shape of [num_boxes]. The data type is float32 or float64. Default: None.
        category_idxs(Tensor, optional): Category indices corresponding to boxes. 
            it's a 1D-Tensor with shape of [num_boxes]. The data type is int64. Default: None.
        categories(List, optional): A list of unique id of all categories. The data type is int64. Default: None.
        top_k(int64, optional): The top K boxes who has higher score and kept by NMS preds to 
            consider. top_k should be smaller equal than num_boxes. Default: None.

    Returns:
        Tensor: 1D-Tensor with the shape of [num_boxes]. Indices of boxes kept by NMS.

    Examples:
        .. code-block:: python
        
            import paddle
            import numpy as np

            boxes = np.random.rand(4, 4).astype('float32')
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            # [[0.06287421 0.5809351  0.3443958  0.8713329 ]
            #  [0.0749094  0.9713205  0.99241287 1.2799143 ]
            #  [0.46246734 0.6753201  1.346266   1.3821303 ]
            #  [0.8984796  0.5619834  1.1254641  1.0201943 ]]

            out =  paddle.vision.ops.nms(paddle.to_tensor(boxes), 0.1)
            # [0, 1, 3, 0]

            scores = np.random.rand(4).astype('float32')
            # [0.98015213 0.3156527  0.8199343  0.874901 ]

            categories = [0, 1, 2, 3]
            category_idxs = np.random.choice(categories, 4)                        
            # [2 0 0 3]

            out =  paddle.vision.ops.nms(paddle.to_tensor(boxes), 
                                                    0.1, 
                                                    paddle.to_tensor(scores), 
                                                    paddle.to_tensor(category_idxs), 
                                                    categories, 
                                                    4)
            # [0, 3, 2]
    """

    def _nms(boxes, iou_threshold):
        if _non_static_mode():
            return _C_ops.nms(boxes, 'iou_threshold', iou_threshold)

        helper = LayerHelper('nms', **locals())
        out = helper.create_variable_for_type_inference('int64')
        helper.append_op(
            type='nms',
            inputs={'Boxes': boxes},
            outputs={'KeepBoxesIdxs': out},
            attrs={'iou_threshold': iou_threshold})
        return out

    if scores is None:
        return _nms(boxes, iou_threshold)

    import paddle
    if category_idxs is None:
        sorted_global_indices = paddle.argsort(scores, descending=True)
        return _nms(boxes[sorted_global_indices], iou_threshold)

    if top_k is not None:
        assert top_k <= scores.shape[
            0], "top_k should be smaller equal than the number of boxes"
    assert categories is not None, "if category_idxs is given, categories which is a list of unique id of all categories is necessary"

    mask = paddle.zeros_like(scores, dtype=paddle.int32)

    for category_id in categories:
        cur_category_boxes_idxs = paddle.where(category_idxs == category_id)[0]
        shape = cur_category_boxes_idxs.shape[0]
        cur_category_boxes_idxs = paddle.reshape(cur_category_boxes_idxs,
                                                 [shape])
        if shape == 0:
            continue
        elif shape == 1:
            mask[cur_category_boxes_idxs] = 1
            continue
        cur_category_boxes = boxes[cur_category_boxes_idxs]
        cur_category_scores = scores[cur_category_boxes_idxs]
        cur_category_sorted_indices = paddle.argsort(
            cur_category_scores, descending=True)
        cur_category_sorted_boxes = cur_category_boxes[
            cur_category_sorted_indices]

        cur_category_keep_boxes_sub_idxs = cur_category_sorted_indices[_nms(
            cur_category_sorted_boxes, iou_threshold)]

        updates = paddle.ones_like(
            cur_category_boxes_idxs[cur_category_keep_boxes_sub_idxs],
            dtype=paddle.int32)
        mask = paddle.scatter(
            mask,
            cur_category_boxes_idxs[cur_category_keep_boxes_sub_idxs],
            updates,
            overwrite=True)
    keep_boxes_idxs = paddle.where(mask)[0]
    shape = keep_boxes_idxs.shape[0]
    keep_boxes_idxs = paddle.reshape(keep_boxes_idxs, [shape])
    sorted_sub_indices = paddle.argsort(
        scores[keep_boxes_idxs], descending=True)

    if top_k is None:
        return keep_boxes_idxs[sorted_sub_indices]

    if _non_static_mode():
        top_k = shape if shape < top_k else top_k
        _, topk_sub_indices = paddle.topk(scores[keep_boxes_idxs], top_k)
        return keep_boxes_idxs[topk_sub_indices]

    return keep_boxes_idxs[sorted_sub_indices][:top_k]
