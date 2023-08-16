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

import warnings

import paddle
from paddle import _C_ops, in_dynamic_mode
from paddle.fluid.layer_helper import LayerHelper
from paddle.framework import no_grad
from paddle.nn.layer.norm import _BatchNormBase


class BatchNorm(paddle.nn.BatchNorm1D):
    r"""
    Applies Batch Normalization over a SparseCooTensor as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .

    When use_global_stats = False, the :math:`\mu_{\beta}`
    and :math:`\sigma_{\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:

    ..  math::

        \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i \qquad &//\
        \ mini-batch\ mean \\
        \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \
        \mu_{\beta})^2 \qquad &//\ mini-batch\ variance \\

    When use_global_stats = True, the :math:`\mu_{\beta}`
    and :math:`\sigma_{\beta}^{2}` are not the statistics of one mini-batch.
    They are global or running statistics (moving_mean and moving_variance). It usually got from the
    pre-trained model. Calculated as follows:

    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global \ mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global \ variance \\

    The normalization function formula is as follows:

    ..  math::

        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

    - :math:`\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\gamma` : trainable proportional parameter
    - :math:`\beta` : trainable deviation parameter

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to False, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to False, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, may be "NC", "NCL" or "NLC". Default "NCL".
        use_global_stats(bool|None, optional): Whether to use global mean and variance. If set to False, use the statistics of one mini-batch, if set to True, use the global statistics, if set to None, use global statistics in the test phase and use the statistics of one mini-batch in the training phase. Default: None.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: A SparseCooTensor with layout = 'NDHWC' or 'NHWC'.
        - output: SparseCooTensor with same shape as input x.

    Returns:
        None.


    Examples:
        .. code-block:: python

          import paddle

          paddle.seed(123)
          channels = 3
          x_data = paddle.randn((1, 6, 6, 6, channels)).astype('float32')
          dense_x = paddle.to_tensor(x_data)
          sparse_x = dense_x.to_sparse_coo(4)
          batch_norm = paddle.sparse.nn.BatchNorm(channels)
          batch_norm_out = batch_norm(sparse_x)
          print(batch_norm_out.shape)
          # [1, 6, 6, 6, 3]
    """

    def __init__(
        self,
        num_features,
        momentum=0.9,
        epsilon=1e-05,
        weight_attr=None,
        bias_attr=None,
        data_format='NDHWC',
        use_global_stats=None,
        name=None,
    ):
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=epsilon,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format,
            use_global_stats=use_global_stats,
            name=name,
        )

    def _check_data_format(self, input):
        if input not in ["NDHWC", "NHWC"]:
            raise ValueError(
                'sparse BatchNorm only support layout of "NDHWC" and "NHWC"'
            )

    def forward(self, input):
        self._check_data_format(self._data_format)

        if self.training:
            warnings.warn(
                "When training, we now always track global mean and variance."
            )

        if self._use_global_stats is None:
            self._use_global_stats = not self.training
            trainable_statistics = False
        else:
            trainable_statistics = not self._use_global_stats

        data_format = 'NCHW' if self._data_format[1] == 'C' else 'NHWC'

        if in_dynamic_mode():
            batch_norm_out, _, _, _, _, _ = _C_ops.sparse_batch_norm_(
                input,
                self._mean,
                self._variance,
                self.weight,
                self.bias,
                not self.training,
                self._momentum,
                self._epsilon,
                data_format,
                self._use_global_stats,
                trainable_statistics,
            )
            return batch_norm_out
        else:
            inputs = {
                'x': input,
                'scale': self.weight,
                'bias': self.bias,
                'mean': self._mean,
                'variance': self._variance,
            }
            attrs = {
                'momentum': self._momentum,
                'epsilon': self._epsilon,
                'data_layout': data_format,
                'is_test': not self.training,
                'use_global_stats': self._use_global_stats,
                'trainable_statistics': trainable_statistics,
                'fuse_with_relu': False,
            }
            op_type = 'sparse_batch_norm'
            helper = LayerHelper(op_type)
            dtype = input.dtype
            mean_out = helper.create_variable_for_type_inference(
                dtype=dtype, stop_gradient=True
            )
            variance_out = helper.create_variable_for_type_inference(
                dtype=dtype, stop_gradient=True
            )
            saved_mean = helper.create_variable_for_type_inference(
                dtype=dtype, stop_gradient=True
            )
            saved_variance = helper.create_variable_for_type_inference(
                dtype=dtype, stop_gradient=True
            )
            reserve_space = helper.create_variable_for_type_inference(
                dtype=dtype, stop_gradient=True
            )
            out = helper.create_sparse_variable_for_type_inference(dtype)
            outputs = {
                "out": out,
                "mean_out": mean_out,
                "variance_out": variance_out,
                "saved_mean": saved_mean,
                "saved_variance": saved_variance,
                "reserve_space": reserve_space,
            }
            helper.append_op(
                type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
            )
            return out


class SyncBatchNorm(paddle.nn.SyncBatchNorm):
    r"""
    This interface is used to construct a callable object of the ``SyncBatchNorm`` class.
    It implements the function of the Cross-GPU Synchronized Batch Normalization Layer, and can
    be used as a normalizer function for other operations, such as conv2d and fully connected
    operations.
    The data is normalized by the mean and variance of the channel based on whole mini-batch
    , which including data in all gpus.
    Refer to `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_
    for more details.

    When model in training mode, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are the statistics of whole mini-batch data in all gpus.
    Calculated as follows:

    ..  math::

        \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i \qquad &//\
        \ mini-batch\ mean \\
        \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \
        \mu_{\beta})^2 \qquad &//\ mini-batch\ variance \\

    - :math:`x` : whole mini-batch data in all gpus
    - :math:`m` : the size of the whole mini-batch data

    When model in evaluation mode, the :math:`\\mu_{\\beta}`
    and :math:`\sigma_{\beta}^{2}` are global statistics (moving_mean and moving_variance,
    which usually got from the pre-trained model). Global statistics calculated as follows:

    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global \ mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global \ variance \\

    The formula of normalization is as follows:

    ..  math::

        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
        \sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift

    - :math:`\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\gamma` : trainable scale parameter vector
    - :math:`\beta` : trainable shift parameter vector

    Note:
        If you want to use container to pack your model and has ``SyncBatchNorm`` in the
        evaluation phase, please use ``nn.LayerList`` or ``nn.Sequential`` instead of
        ``list`` to pack the model.

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
             of this layer. If it is set to None or one attribute of ParamAttr, this layerr
             will create ParamAttr as param_attr. If the Initializer of the param_attr
             is not set, the parameter is initialized with Xavier. If it is set to False,
             this layer will not have trainable scale parameter. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of this layer.
             If it is set to None or one attribute of ParamAttr, this layer
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. If it is set to False, this layer will not
             have trainable bias parameter. Default: None.
        data_format(str, optional): Specify the input data format, may be "NCHW". Default "NCHW".
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shapes:
        input: Tensor that the dimension from 2 to 5.

        output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

          # required: gpu
          import paddle
          import paddle.sparse.nn as nn

          x = paddle.to_tensor([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]], dtype='float32')
          x = x.to_sparse_coo(len(x.shape)-1)

          if paddle.is_compiled_with_cuda():
              sync_batch_norm = nn.SyncBatchNorm(2)
              hidden1 = sync_batch_norm(x)
              print(hidden1)
              # Tensor(shape=[1, 2, 2, 2], dtype=paddle.float32, place=Place(gpu:0), stop_gradient=True,
              #        indices=[[0, 0, 0, 0],
              #                 [0, 0, 1, 1],
              #                 [0, 1, 0, 1]],
              #        values=[[-0.40730840, -0.13725480],
              #                 [-0.40730840, -1.20299828],
              #                 [ 1.69877410, -0.23414057],
              #                 [-0.88415730,  1.57439375]])
    """

    def __init__(
        self,
        num_features,
        momentum=0.9,
        epsilon=1e-05,
        weight_attr=None,
        bias_attr=None,
        data_format='NCHW',
        name=None,
    ):
        super().__init__(
            num_features,
            momentum,
            epsilon,
            weight_attr,
            bias_attr,
            data_format,
            name,
        )

    def forward(self, x):
        self._check_data_format()
        sync_batch_norm_out, _, _, _, _, _ = _C_ops.sparse_sync_batch_norm_(
            x,
            self._mean,
            self._variance,
            self.weight,
            self.bias,
            not self.training,
            self._momentum,
            self._epsilon,
            self._data_format,
            False,
            False,
        )
        return sync_batch_norm_out

    @classmethod
    def convert_sync_batchnorm(cls, layer):
        r"""
        Helper function to convert :class: `paddle.sparse.nn.BatchNorm` layers in the model to :class: `paddle.sparse.nn.SyncBatchNorm` layers.

        Parameters:
            layer(paddle.nn.Layer): model containing one or more `BatchNorm` layers.

        Returns:
            The original model with converted SyncBatchNorm layers. If BatchNorm layer in the model, use SyncBatchNorm layer instead.

        Examples:

            .. code-block:: python

                import paddle
                import paddle.sparse.nn as nn

                model = paddle.nn.Sequential(nn.Conv3D(3, 5, 3), nn.BatchNorm(5))
                sync_model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        """

        layer_output = layer
        if isinstance(layer, _BatchNormBase):
            if (
                layer._weight_attr is not None
                and not isinstance(layer._weight_attr, bool)
                and layer._weight_attr.name is not None
            ):
                layer._weight_attr.name = layer._weight_attr.name + '_sync'
            if (
                layer._bias_attr is not None
                and not isinstance(layer._bias_attr, bool)
                and layer._bias_attr.name is not None
            ):
                layer._bias_attr.name = layer._bias_attr.name + '_sync'

            # convert sparse BatchNorm
            if isinstance(layer, BatchNorm):
                layer_output = SyncBatchNorm(
                    layer._num_features,
                    layer._momentum,
                    layer._epsilon,
                    layer._weight_attr,
                    layer._bias_attr,
                    layer._data_format,
                    layer._name,
                )
            # convert dense BatchNorm
            else:
                layer_output = paddle.nn.SyncBatchNorm(
                    layer._num_features,
                    layer._momentum,
                    layer._epsilon,
                    layer._weight_attr,
                    layer._bias_attr,
                    layer._data_format,
                    layer._name,
                )

            if (
                layer._weight_attr is not False
                and layer._bias_attr is not False
            ):
                with no_grad():
                    layer_output.weight = layer.weight
                    layer_output.bias = layer.bias
            layer_output._mean = layer._mean
            layer_output._variance = layer._variance

        for name, sublayer in layer.named_children():
            layer_output.add_sublayer(
                name, cls.convert_sync_batchnorm(sublayer)
            )
        del layer
        return layer_output
