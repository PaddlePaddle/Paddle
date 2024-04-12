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

import paddle
from paddle.base.data_feeder import check_type, check_variable_and_dtype
from paddle.base.framework import in_dygraph_mode
from paddle.base.layer_helper import LayerHelper
from paddle.base.layers.layer_function_generator import templatedoc

__all__ = []


@templatedoc()
def sequence_conv(
    input,
    num_filters,
    filter_size=3,
    filter_stride=1,
    padding=True,
    padding_start=None,
    bias_attr=None,
    param_attr=None,
    act=None,
    name=None,
):
    r"""

    Note:
        Only receives Tensor as input. If your input is Tensor, please use conv2d Op.(base.layers.** :ref:`api_paddle_nn_functional_conv2d` ).

    This operator receives input sequences with variable length and other convolutional
    configuration parameters(num_filters, filter_size) to apply the convolution operation.
    It fills all-zero padding data on both sides of the sequence by default to ensure that
    the output is the same length as the input. You can customize the padding behavior by
    configuring the parameter :attr:`padding\_start` .

    **Warning:** the parameter :attr:`padding` take no effect and will be deprecated in the future.

    .. code-block:: text

            Here we will illustrate the details of the padding operation:
            For a mini-batch of 2 variable lengths sentences, containing 3, and 1 time-steps:
            Assumed input (X) is a [4, N] float Tensor, and for the sake of simplicity, we assume N=2.
            input.data = [[1, 1],
                          [2, 2],
                          [3, 3],
                          [4, 4]]

            This is to say that input (X) has 4 words and the dimension of each word
            representation is 2.

            * Case1:

                If padding_start is -1 and filter_size is 3.
                The length of padding data is calculated as follows:
                up_pad_len = max(0, -padding_start) = 1
                down_pad_len = max(0, filter_size + padding_start - 1) = 1

                The output of the input sequence after padding is:
                data_after_padding = [[0, 0, 1, 1, 2, 2],
                                      [1, 1, 2, 2, 3, 3],
                                      [2, 2, 3, 3, 0, 0],
                                      [0, 0, 4, 4, 0, 0]]

                It will be multiplied by the filter weight to get the final output.
                Assume num_filters = 3
                output.data = [[ 0.3234, -0.2334,  0.7433],
                               [ 0.5646,  0.9464, -0.1223],
                               [-0.1343,  0.5653,  0.4555],
                               [ 0.9954, -0.1234, -0.1234]]
                output.shape = [4, 3]     # 3 = num_filters
                output.lod = [[0, 3, 4]]  # Remain the same


    Args:
        input (Tensor): Tensor with shape :math:`(M, K)`, where M is the total time-step of mini-batch
            and K is hidden_size of input. Only lod_level of 1 is supported. The data type should be float32 or
            float64.
        num_filters (int): the number of filters.
        filter_size (int): the height of filter. Specified filter width is not supported, the width is
            hidden_size by default. Default: 3.
        filter_stride (int, optional): stride of the filter. Currently only supports :attr:`stride` = 1.
        padding (bool, optional): the parameter :attr:`padding` take no effect and will be discarded in the
            future. Currently, it will always pad input to make sure the length of the output is
            the same as input whether :attr:`padding` is set true or false. Because the length of
            input sequence may be shorter than :attr:`filter\_size`, which will cause the convolution
            result to not be computed correctly. These padding data will not be trainable or updated
            while training. Default: True.
        padding_start (int): It is used to indicate the start index for padding the input
            sequence, which can be negative. The negative number means to pad
            :attr:`|padding_start|` time-steps of all-zero data at the beginning of each instance.
            The positive number means to skip :attr:`padding_start` time-steps of each instance,
            and it will pad :math:`filter\_size + padding\_start - 1` time-steps of all-zero data
            at the end of the sequence to ensure that the output is the same length as the input.
            If set None, the same length :math:`\\frac{filter\_size}{2}` of data will be filled
            on both sides of the sequence. If set 0, the length of :math:`filter\_size - 1` data
            is padded at the end of each input sequence. Default: None.
        bias_attr (ParamAttr): To specify the bias parameter property. Default: None, which means the
            default bias parameter property is used. See usage for details in :ref:`api_paddle_ParamAttr` .
        param_attr (ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_paddle_ParamAttr` .
        act (str): Activation to be applied to the output of this layer, such as tanh, softmax,
            sigmoid, relu. For more information, please refer to :ref:`api_guide_activations_en` . Default: None.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: Tensor with the same length as input. The data type is float32 or float64, which is same as input.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> x = paddle.static.data(name='x', shape=[-1, 10], dtype='float32', lod_level=1)
            >>> x_conved = paddle.static.nn.sequence_conv(input=x, num_filters=2, filter_size=3, padding_start=-1)
    """

    assert (
        not in_dygraph_mode()
    ), "sequence layer is not supported in dygraph mode yet."
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'sequence_conv'
    )
    helper = LayerHelper('sequence_conv', **locals())
    dtype = helper.input_dtype()
    filter_shape = [filter_size * input.shape[1], num_filters]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype
    )
    pre_bias = helper.create_variable_for_type_inference(dtype)
    if padding_start is None:
        padding_start = -int(filter_size // 2)

    helper.append_op(
        type='sequence_conv',
        inputs={
            'X': [input],
            'Filter': [filter_param],
        },
        outputs={"Out": pre_bias},
        attrs={
            'contextStride': filter_stride,
            'contextStart': padding_start,
            'contextLength': filter_size,
        },
    )
    pre_act = helper.append_bias_op(pre_bias)
    return helper.append_activation(pre_act)


def sequence_softmax(input, use_cudnn=False, name=None):
    r"""

    Note:
        The input type of the OP must be Tensor. For Tensor, use:** :ref:`api_paddle_nn_functional_softmax`

    A LoD-tensor can be regarded as several sequences, and this op apply softmax algo on each sequence.
    The shape of input Tensor can be :math:`[N, 1]` or :math:`[N]`, where :math:`N`
    is the sum of the length of all sequences. Recommended usage: :math:`[N]`.

    For i-th sequence in a mini-batch:

    .. math::

        Out(X[lod[i]:lod[i+1]], :) = \\frac{\exp(X[lod[i]:lod[i+1], :])}{\sum(\exp(X[lod[i]:lod[i+1], :]))}

    For example, for a LoD-Tensor with 6 sequences ([3, 2, 4, 1, 2, 3] - sequence length list in order),
    the lod in the runtime is [[0, 3, 5, 9, 10, 12, 15]],
    then softmax will be computed among :math:`X[0:3,:],X[3:5,:],X[5:9,:],X[9:10,:],X[10:12,:],X[12:15,:]`,
    and :math:`N` turns out to be 15.

    .. code-block:: text

        *Case 1:

            Given:
                input.data = [0.7, 1, 0.6,
                              1.5, 1.1,
                              1.2, 0.2, 0.6, 1.9,
                              3.1,
                              2.5, 0.8,
                              0.1, 2.4, 1.3]
                input.lod = [[0, 3, 5, 9, 10, 12, 15]]
            then:
                 output.data = [0.30724832, 0.41474187, 0.2780098,
                                0.59868765, 0.40131235,
                                0.2544242, 0.09359743, 0.13963096, 0.5123474,
                                1.,
                                0.84553474, 0.15446526,
                                0.06995796, 0.69777346, 0.23226859]
                 output.lod = [[0, 3, 5, 9, 10, 12, 15]]


    Args:
        input (Tensor):A Tensor with shape of  :math:`[N, 1]` or  :math:`[N]`, Recommended usage: :math:`[N]`.
                         Supported data types: float32, float64.
        use_cudnn (bool, optional): Use cudnn kernel or not. Effective only when the cudnn version of the paddle
                                    library is installed and GPU is used for training or reasoning. Default: False.
        name (str, optional): The default value is None. Normally there is no need for user to set this property.
                              For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: A LoD-Tensor which has the same shape and data type with input.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> x = paddle.static.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
            >>> x_sequence_softmax_1 = paddle.static.nn.sequence_softmax(input=x)

            >>> y = paddle.static.data(name='y', shape=[7],
            ...     dtype='float32', lod_level=1)
            >>> x_sequence_softmax_2 = paddle.static.nn.sequence_softmax(input=y)
    """
    assert (
        not in_dygraph_mode()
    ), "sequence layer is not supported in dygraph mode yet."
    helper = LayerHelper('sequence_softmax', **locals())
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'sequence_softmax'
    )
    dtype = helper.input_dtype()
    softmax_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="sequence_softmax",
        inputs={"X": input},
        outputs={"Out": softmax_out},
        attrs={"use_cudnn": use_cudnn},
    )
    return softmax_out


def sequence_pool(input, pool_type, is_test=False, pad_value=0.0):
    r"""

    Note:
        Only receives Tensor as input. If your input is Tensor, please use pool2d Op.(static.nn.** :ref:`api_paddle_nn_functional_avg_pool2d` or :ref:`api_paddle_nn_functional_max_pool2d` ).

    This operator only supports Tensor as input. It will apply specified pooling
    operation on the input Tensor. It pools features of all time-steps of each
    sequence at the last lod_level using :attr:`pool_type` mentioned in the parameters,
    such as sum, average, sqrt, etc.

    It supports six pool_type:

    - average: :math:`Out[i] = \\frac{\sum_i X_i}{N}`
    - sum:     :math:`Out[i] = \sum_jX_{ij}`
    - sqrt:    :math:`Out[i] = \\frac{\sum_jX_{ij}}{\sqrt{len(X_i)}}`
    - max:     :math:`Out[i] = max(X_i)`
    - last:    :math:`Out[i] = X_{N_i}`
    - first:   :math:`Out[i]` = X_0

    where :math:`N_i` is the length of i-th input sequence.

    .. code-block:: text

        Case 1:
        input is a 1-level Tensor and pad_value = 0.0:
            input.lod = [[0, 2, 5, 7, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        output is Tensor:
            out.shape = [4, 1]
            with condition out.shape[0] == len(x.lod[-1]) == 4

        for different pool_type:
            average: out.data = [[2.], [4.], [3.], [0.0]], where 2.=(1. + 3.)/2, 4.=(2. + 4. + 6.)/3, 3.=(5. + 1.)/2
            sum    : out.data = [[4.], [12.], [6.], [0.0]], where 4.=1. + 3., 12.=2. + 4. + 6., 6.=5. + 1.
            sqrt   : out.data = [[2.82], [6.93], [4.24], [0.0]], where 2.82=(1. + 3.)/sqrt(2), 6.93=(2. + 4. + 6.)/sqrt(3), 4.24=(5. + 1.)/sqrt(2)
            max    : out.data = [[3.], [6.], [5.], [0.0]], where 3.=max(1., 3.), 6.=max(2., 4., 6.), 5.=max(5., 1.)
            last   : out.data = [[3.], [6.], [1.], [0.0]], where 3.=last(1., 3.), 6.=last(2., 4., 6.), 1.=last(5., 1.)
            first  : out.data = [[1.], [2.], [5.], [0.0]], where 1.=first(1., 3.), 2.=first(2., 4., 6.), 5.=first(5., 1.)

            and all above [0.0] at last of out.data is padding data.

        Case 2:
        input is a 2-level Tensor containing 3 sequences with length info [2, 0, 3],
        where 0 means empty sequence.
        The first sequence contains 2 subsequence with length info [1, 2];
        The last sequence contains 3 subsequence with length info [1, 0, 3].
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        If pool_typ = sum, it will apply pooling on last lod_level [0, 1, 3, 4, 4, 7]. pad_value = 0.0
        output is Tensor:
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            where out.shape[0] == len(x.lod[-1]) == 5
            sum: out.data = [[1.], [5.], [4.], [0.0], [12.]]
            where 1.=1., 5.=3. + 2., 4.=4., 0.0=pad_value, 12.=6. + 5. + 1.

    Args:
        input (variable): Tensor with lod_level no more than 2. The data type should be float32 or float64.
        pool_type (str): The pooling type that supports average, sum, sqrt, max, last or first.
        is_test (bool): Only works when :attr:`pool_type` is max. If set False, a temporary Tensor maxIndex is
            created to record the index information corresponding to the maximum value, which is used for backward
            gradient calculation in the training phase. Default: False.
        pad_value (float): Used to pad the pooling result for empty input sequence. Default: 0.0

    Returns:
        Tensor: Tensor after pooling with data type float32 or float64.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> x = paddle.static.data(name='x', shape=[None, 10], dtype='float32', lod_level=1)
            >>> avg_x = paddle.static.nn.sequence_pool(input=x, pool_type='average')
            >>> sum_x = paddle.static.nn.sequence_pool(input=x, pool_type='sum')
            >>> sqrt_x = paddle.static.nn.sequence_pool(input=x, pool_type='sqrt')
            >>> max_x = paddle.static.nn.sequence_pool(input=x, pool_type='max')
            >>> last_x = paddle.static.nn.sequence_pool(input=x, pool_type='last')
            >>> first_x = paddle.static.nn.sequence_pool(input=x, pool_type='first')
    """
    assert (
        not in_dygraph_mode()
    ), "sequence layer is not supported in dygraph mode yet."
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'sequence_pool'
    )
    helper = LayerHelper('sequence_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    max_index = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type="sequence_pool",
        inputs={"X": input},
        outputs={"Out": pool_out, "MaxIndex": max_index},
        attrs={
            "pooltype": pool_type.upper(),
            "is_test": is_test,
            "pad_value": pad_value,
        },
    )

    # when pool_type is max, variable max_index is initialized,
    # so we stop the gradient explicitly here
    if pool_type == 'max':
        max_index.stop_gradient = True

    return pool_out


@templatedoc()
def sequence_concat(input, name=None):
    """

    Note:
        Only receives Tensor as input. If your input is Tensor, please use concat Op.(static.nn.** :ref:`api_paddle_concat` ).

    This operator only supports Tensor as input. It concatenates the multiple Tensor from input by the LoD information,
    and outputs the concatenated Tensor.

    .. code-block:: text

        input is a list of Tensor:
            input = [x1, x2]
        where:
            x1.lod = [[0, 3, 5]]
            x1.data = [[1], [2], [3], [4], [5]]
            x1.shape = [5, 1]

            x2.lod = [[0, 2, 4]]
            x2.data = [[6], [7], [8], [9]]
            x2.shape = [4, 1]
        and should satisfy: len(x1.lod[0]) == len(x2.lod[0])

        output is Tensor:
            out.lod = [[0, 3+2, 5+4]]
            out.data = [[1], [2], [3], [6], [7], [4], [5], [8], [9]]
            out.shape = [9, 1]

    Args:
        input(list of Tensor): List of Tensor to be concatenated. The length of each Tensor should be same.
            The data type can be float32, float64 or int64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: Output the concatenated Tensor. The data type is same as input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> x = paddle.static.data(name='x', shape=[-1, 10], dtype='float32', lod_level=1)
            >>> y = paddle.static.data(name='y', shape=[-1, 10], dtype='float32', lod_level=1)
            >>> out = paddle.static.nn.sequence_concat(input=[x, y])
    """
    assert (
        not in_dygraph_mode()
    ), "sequence layer is not supported in dygraph mode yet."
    helper = LayerHelper('sequence_concat', **locals())

    check_type(
        input, 'input', list, 'paddle.static.nn.sequence_lod.sequence_concat'
    )
    for i, input_x in enumerate(input):
        check_variable_and_dtype(
            input_x,
            'input[' + str(i) + ']',
            ['int64', 'float32', 'float64'],
            'paddle.static.nn.sequence_lod.sequence_concat',
        )

    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='sequence_concat', inputs={'X': input}, outputs={'Out': [out]}
    )
    return out


def sequence_first_step(input):
    """

    Only supports Tensor as input. Given the input Tensor, it will
    select first time-step feature of each sequence as output.

    .. code-block:: text

       Case 1:
        input is 1-level Tensor:
            input.lod = [[0, 2, 5, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        output is a Tensor:
            out.shape = [3, 1]
            out.shape[0] == len(x.lod[-1]) == 3
            out.data = [[1.], [2.], [5.]], where 1.=first(1., 3.), 2.=first(2., 4., 6.), 5.=first(5., 1.)

        Case 2:
        input is a 2-level Tensor containing 3 sequences with length info [2, 0, 3],
        where 0 means empty sequence.
        The first sequence contains 2 subsequence with length info [1, 2];
        The last sequence contains 3 subsequence with length info [1, 0, 3].
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        It will apply pooling on last lod_level [0, 1, 3, 4, 4, 7]. pad_value = 0.0
        output is a Tensor:
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            out.shape[0] == len(x.lod[-1]) == 5
            out.data = [[1.], [3.], [4.], [0.0], [6.]]
            where 1.=first(1.), 3.=first(3., 2.), 4.=first(4.), 0.0 = pad_value, 6.=first(6., 5., 1.)

    Args:
        input(Tensor): Tensor with lod_level no more than 2. The data type should be float32 or float64.

    Returns:
        Tensor: Tensor consist of the sequence's first step vector. The data type is float32 or float64.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> x = paddle.static.data(name='x', shape=[None, 10], dtype='float32', lod_level=1)
            >>> x_first_step = paddle.static.nn.sequence_first_step(input=x)
    """
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'sequence_first_step'
    )
    return sequence_pool(input=input, pool_type="first")


def sequence_last_step(input):
    """

    Only supports Tensor as input. Given the input Tensor, it will
    select last time-step feature of each sequence as output.

    .. code-block:: text

        Case 1:
        input is 1-level Tensor:
            input.lod = [[0, 2, 5, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        output is a Tensor:
            out.shape = [3, 1]
            out.shape[0] == len(x.lod[-1]) == 3
            out.data = [[3.], [6.], [1.]], where 3.=last(1., 3.), 6.=last(2., 4., 6.), 1.=last(5., 1.)

        Case 2:
        input is a 2-level Tensor containing 3 sequences with length info [2, 0, 3],
        where 0 means empty sequence.
        The first sequence contains 2 subsequence with length info [1, 2];
        The last sequence contains 3 subsequence with length info [1, 0, 3].
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        It will apply pooling on last lod_level [0, 1, 3, 4, 4, 7]. pad_value = 0.0
        output is a Tensor:
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            out.shape[0] == len(x.lod[-1]) == 5
            out.data = [[1.], [2.], [4.], [0.0], [1.]]
            where 1.=last(1.), 2.=last(3., 2.), 4.=last(4.), 0.0 = pad_value, 1=last(6., 5., 1.)


    Args:
        input(Tensor): Tensor with lod_level no more than 2. The data type should be float32.

    Returns:
        Tensor: Tensor consist of the sequence's last step vector. The data type is float32.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> x = paddle.static.data(name='x', shape=[None, 10], dtype='float32', lod_level=1)
            >>> x_last_step = paddle.static.nn.sequence_last_step(input=x)
    """
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'sequence_last_step'
    )
    return sequence_pool(input=input, pool_type="last")


def sequence_mask(x, maxlen=None, dtype='int64', name=None):
    r"""
    **SequenceMask Layer**

    This layer outputs a mask according to the input :code:`x` and
    :code:`maxlen` with data type of :code:`dtype`.

    Supposing :code:`x` is a Tensor with shape [d_1, d_2, ..., d_n], the
    :code:`y` is a mask with shape [d_1, d_2, ..., d_n, maxlen], where:

    .. math::

        y(i_1, i_2,..., i_n, j) = (j < x(i_1, i_2,..., i_n))

    .. code-block:: text

        Case:

        Consider input:
            x = [3, 1, 1, 0]    max_len = 4

        then we get out:
            mask = [[1, 1, 1, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0]]

    Args:
        x (Tensor): Input tensor of sequence_mask layer, \
            whose elements are integers less than :code:`maxlen`. \
            Tensor or Tensor with shape [d_1, d_2, ..., d_n].
        maxlen (int, optional): Maximum length of the sequence. If :code:`maxlen` \
                           is None, it would be replace with :math:`max(x)`.
        dtype (np.dtype|paddle.dtype|str, optional): Data type of the output, \
             ``int64`` by default.
        name(str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns:
            Tensor, The output sequence mask. Tensor with shape [d_1, d_2, ..., d_n, maxlen]
            and data type of :code:`dtype`. The data type should be bool, float32, float64, int8,
            int32 or int64.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> lengths = paddle.to_tensor([10, 9, 8])
            >>> mask = paddle.nn.functional.sequence_mask(lengths)

            >>> print(mask.numpy())
            [[1 1 1 1 1 1 1 1 1 1]
             [1 1 1 1 1 1 1 1 1 0]
             [1 1 1 1 1 1 1 1 0 0]]

    """

    return paddle.nn.functional.sequence_mask(x, maxlen, dtype, name)
