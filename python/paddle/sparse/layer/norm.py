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

import paddle
import warnings


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
            will create ParamAttr as weight_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, may be "NC", "NCL" or "NLC". Defalut "NCL".
        use_global_stats(bool|None, optional): Whether to use global mean and variance. If set to False, use the statistics of one mini-batch, if set to True, use the global statistics, if set to None, use global statistics in the test phase and use the statistics of one mini-batch in the training phase. Default: None.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Shape:
        - x: A SparseCooTensor with layout = 'NDHWC'.
        - output: SparseCooTensor with same shape as input x.

    Returns:
        None.
    

    Examples:
        .. code-block:: python

          import paddle
          from paddle.fluid.framework import _test_eager_guard

          with _test_eager_guard():
              paddle.seed(123)
              channels = 3
              x_data = paddle.randn((1, 6, 6, 6, channels)).astype('float32')
              dense_x = paddle.to_tensor(x_data) 
              sparse_x = dense_x.to_sparse_coo(4)
              batch_norm = paddle.sparse.BatchNorm(channels)
              batch_norm_out = batch_norm(sparse_x)
              print(batch_norm_out.shape)
              # [1, 6, 6, 6, 3]
    """

    def __init__(self,
                 num_features,
                 momentum=0.9,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NDHWC',
                 use_global_stats=None,
                 name=None):
        super(BatchNorm, self).__init__(
            num_features,
            momentum=momentum,
            epsilon=epsilon,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format,
            use_global_stats=use_global_stats,
            name=name)

    def _check_data_format(self, input):
        if input != "NDHWC":
            raise ValueError('sparse BatchNorm only support layout of "NDHWC"')

    def forward(self, input):
        values = input.values()
        self._check_data_format(self._data_format)

        if len(values.shape) != 2:
            raise ValueError('expected 2D input.values() (got {}D)'.format(
                len(values.shape)))

        if self.training:
            warnings.warn(
                "When training, we now always track global mean and variance.")

        batch_norm_out = paddle.nn.functional.batch_norm(
            values,
            self._mean,
            self._variance,
            weight=self.weight,
            bias=self.bias,
            training=self.training,
            momentum=self._momentum,
            epsilon=self._epsilon,
            data_format='NC',
            use_global_stats=self._use_global_stats)

        return paddle.sparse.sparse_coo_tensor(
            input.indices(),
            batch_norm_out,
            shape=input.shape,
            stop_gradient=input.stop_gradient)
