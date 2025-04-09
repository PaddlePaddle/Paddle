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

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import paddle
from paddle import base, in_dynamic_mode
from paddle.base.framework import in_dynamic_or_pir_mode

from .. import functional as F
from .layers import Layer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle._typing import ParamAttrLike

    from ..functional.loss import _ReduceMode


__all__ = []


class BCEWithLogitsLoss(Layer):
    r"""

    Combine the sigmoid layer and the :ref:`api_paddle_nn_BCELoss` layer.

    This measures the element-wise probability error in classification tasks
    in which each class is independent.
    This can be thought of as predicting labels for a data-point, where labels
    are not mutually exclusive. For example, a news article can be about
    politics, technology or sports at the same time or none of these.

    Firstly, calculate loss function as follows:

    .. math::
           Out = -Labels * \log(\sigma(Logit)) - (1 - Labels) * \log(1 - \sigma(Logit))

    We know that :math:`\sigma(Logit) = \frac{1}{1 + e^{-Logit}}`. By substituting this we get:

    .. math::
           Out = Logit - Logit * Labels + \log(1 + e^{-Logit})

    For stability and to prevent overflow of :math:`e^{-Logit}` when Logit < 0,
    we reformulate the loss as follows:

        .. math::
           Out = \max(Logit, 0) - Logit * Labels + \log(1 + e^{-\|Logit\|})

    Then, if ``weight`` or ``pos_weight`` is not None, then multiply the
    weight tensor on the loss `Out`. The ``weight`` tensor will attach different
    weight on every items in the batch. The ``pos_weight`` will attach different
    weight on the positive label of each class.

    Finally, apply reduce operation on the loss.
    If :attr:`reduction` set to ``'none'``, will return the original loss `Out`.
    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is :math:`Out = MEAN(Out)`.
    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is :math:`Out = SUM(Out)`.

    Note that the target labels ``label`` should be numbers between 0 and 1.

    Args:
        weight (Tensor|None, optional): A manual rescaling weight given to the loss of each
            batch element. If given, it has to be a 1D Tensor whose size is `[N, ]`,
            The data type is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.
        pos_weight (Tensor|None, optional): A weight of positive examples. Must be a vector
            with length equal to the number of classes. The data type is float32, float64.
            Default is ``'None'``.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shapes:
        - logit (Tensor): The input predications tensor. 2-D tensor with shape: [N, `*`], N is batch_size, `*` means number of additional dimensions. The ``logit`` is usually the output of Linear layer. Available dtype is float32, float64.
        - label (Tensor): The target labels tensor. 2-D tensor with the same shape as ``logit``. The target labels which values should be numbers between 0 and 1. Available dtype is float32, float64.
        - output (Tensor): If ``reduction`` is ``'none'``, the shape of output is same as ``logit`` , else the shape of output is scalar.

    Returns:
        A callable object of BCEWithLogitsLoss.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> logit = paddle.to_tensor([5.0, 1.0, 3.0], dtype="float32")
            >>> label = paddle.to_tensor([1.0, 0.0, 1.0], dtype="float32")
            >>> bce_logit_loss = paddle.nn.BCEWithLogitsLoss()
            >>> output = bce_logit_loss(logit, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.45618808)

    """

    weight: Tensor | None
    reduction: _ReduceMode
    pos_weight: Tensor | None
    name: str | None

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: _ReduceMode = 'mean',
        pos_weight: Tensor | None = None,
        name: str | None = None,
    ) -> None:
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in BCEWithLogitsLoss should be 'sum', 'mean' or 'none', but "
                f"received {reduction}, which is not allowed."
            )

        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.name = name

    def forward(self, logit: Tensor, label: Tensor) -> Tensor:
        out = paddle.nn.functional.binary_cross_entropy_with_logits(
            logit,
            label,
            self.weight,
            self.reduction,
            self.pos_weight,
            self.name,
        )
        return out


class CrossEntropyLoss(Layer):
    r"""

    By default, the cross entropy loss function is implemented using softmax. This function
    combines the calculation of the softmax operation and the cross entropy loss function
    to provide a more numerically stable computing.

    Calculate the cross entropy loss function without softmax when use_softmax=False.

    By default, calculate the mean of the result, and you can also affect
    the default behavior by using the reduction parameter. Please refer to the part of
    parameters for details.

    Can be used to calculate the softmax cross entropy loss with soft and hard labels.
    Where, the hard labels mean the actual label value, 0, 1, 2, etc.  And the soft labels
    mean the probability of the actual label, 0.6, 0.8, 0.2, etc.

    The calculation includes the following two steps.

    -  **I.softmax cross entropy**

        1. Hard label (each sample can only be assigned into one category)

        1.1. when use_softmax=True

            .. math::
              \\loss_j=-\text{logits}_{label_j}+\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories.

        1.2. when use_softmax=False

            .. math::
              \\loss_j=-\log\left({P}_{label_j}\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories, P is input(the output of softmax).


        2. Soft label (each sample is assigned to multiple categories with a certain probability, and the probability sum is 1).

        2.1. when use_softmax=True

            .. math::
              \\loss_j=-\sum_{i=0}^{C}\text{label}_i\left(\text{logits}_i-\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories.

        2.2. when use_softmax=False

            .. math::
              \\loss_j=-\sum_{j=0}^{C}\left({label}_j*\log\left({P}_{label_j}\right)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories, P is input(the output of softmax).



    -  **II.Weight and reduction processing**

        1. Weight

            If the ``weight`` parameter is ``None`` , go to the next step directly.

            If the ``weight`` parameter is not ``None`` , the cross entropy of each sample is weighted by weight
            according to soft_label = False or True as follows.

            1.1. Hard labels (soft_label = False)

            .. math::
                \\loss_j=loss_j*weight[label_j]


            1.2. Soft labels (soft_label = True)

             .. math::
                \\loss_j=loss_j*\sum_{i}\left(weight[label_i]*logits_i\right)

        2. reduction

            2.1 if the ``reduction`` parameter is ``none``

            Return the previous result directly

            2.2 if the ``reduction`` parameter is ``sum``

            Return the sum of the previous results

            .. math::
               \\loss=\sum_{j}loss_j

            2.3 if the ``reduction`` parameter is ``mean`` , it will be processed according to
            the ``weight`` parameter as follows.

            2.3.1. If the  ``weight``  parameter is ``None``

            Return the average value of the previous results

             .. math::
                \\loss=\sum_{j}loss_j/N

            where, N is the number of samples and C is the number of categories.

            2.3.2. If the ``weight`` parameter is ``None`` , the weighted average value of the previous result will be returned

            1. Hard labels (soft_label = False)

             .. math::
                \\loss=\sum_{j}loss_j/\sum_{j}weight[label_j]

            2. Soft labels (soft_label = True)

             .. math::
                \\loss=\sum_{j}loss_j/\sum_{j}\left(\sum_{i}weight[label_i]\right)


    Parameters:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size C and the data type is float32, float64.
            Default is ``'None'`` .
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the loss. A negative value means that no label
            value needs to be ignored. Only valid when soft_label = False.
            Default is ``-100`` .
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        soft_label (bool, optional): Indicate whether label is soft.
            If soft_label=False, the label is hard.  If soft_label=True, the label is soft.
            Default is ``False``.
        label_smoothing (float, optional): A float in [0.0, 1.0].
            Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing.
            The targets become  a mixture of the original ground truth and a uniform distribution as
            described in paper 'Rethinking the Inception Architecture for Computer Vision'.
            Default is ``0.0``.
        axis (int, optional): The index of dimension to perform softmax calculations.
            It should be in range :math:`[-1, rank - 1]`, where :math:`rank` is the number
            of dimensions of input :attr:`input`.
            Default is ``-1`` .
        use_softmax (bool, optional): Indicate whether compute softmax before cross_entropy.
            Default is ``True``.
        name (str|None, optional): The name of the operator. Default is ``None`` .
            For more information, please refer to :ref:`api_guide_Name` .


    Shape:
        - **input** (Tensor), the data type is float32, float64. Shape is :math:`[N_1, N_2, ..., N_k, C]`, where C is number of classes, ``k >= 1`` .

            Note:

                1. when use_softmax=True, it expects unscaled logits. This operator should not be used with the
                output of softmax operator, which will produce incorrect results.

                2. when use_softmax=False, it expects the output of softmax operator.

        - **label** (Tensor)

            1. If soft_label=False, the shape is
            :math:`[N_1, N_2, ..., N_k]` or :math:`[N_1, N_2, ..., N_k, 1]`, k >= 1.
            the data type is int32, int64, float32, float64, where each value is [0, C-1].

            2. If soft_label=True and no label_smoothing, the shape and data type
            should be same with ``input`` , and the sum of the labels for each sample should be 1.

            3. If has label_smoothing, (i.e. label_smoothing > 0.0), no matter what ``soft_label`` is,
            the shape and data type of ``label`` could be either the situation 1 or situation 2.
            In other words, if label_smoothing > 0.0, the format of label could be one-hot label or integer label.

        - **output** (Tensor), Return the softmax cross_entropy loss of ``input`` and ``label``.
          The data type is the same as input.
          If :attr:`reduction` is ``'mean'`` or ``'sum'`` , the dimension of return value is ``1``.
          If :attr:`reduction` is ``'none'``:

            1. If soft_label = False, the dimension of return value is the same with ``label`` .

            2. if soft_label = True, the dimension of return value is :math:`[N_1, N_2, ..., N_k, 1]` .

    Examples:

        .. code-block:: python
            :name: code-example1

            >>> # hard labels
            >>> import paddle
            >>> paddle.seed(2023)
            >>> N=100
            >>> C=200
            >>> reduction='mean'
            >>> input =  paddle.rand([N, C], dtype='float64')
            >>> label =  paddle.randint(0, C, shape=[N], dtype='int64')
            >>> weight = paddle.rand([C], dtype='float64')

            >>> cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
            ...     weight=weight, reduction=reduction)
            >>> dy_ret = cross_entropy_loss(input, label)
            >>> print(dy_ret)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            5.33697682)

        .. code-block:: python
            :name: code-example2

            >>> # soft labels
            >>> import paddle
            >>> from typing import Optional
            >>> paddle.seed(2023)
            >>> axis = -1
            >>> N = 4
            >>> C = 3
            >>> shape = [N, C]
            >>> reduction='mean'
            >>> weight: Optional[paddle.Tensor] = None
            >>> logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
            >>> # case1: soft labels without label_smoothing
            >>> labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
            >>> labels /= paddle.sum(labels, axis=axis, keepdim=True)
            >>> cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
            ...     weight=weight, reduction=reduction, soft_label=True, label_smoothing=0.0)
            >>> dy_ret = cross_entropy_loss(logits, labels)
            >>> print(dy_ret)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            1.14554912)


            >>> # case2: soft labels with label_smoothing
            >>> import paddle
            >>> from typing import Optional
            >>> paddle.seed(2023)
            >>> axis = -1
            >>> N = 4
            >>> C = 3
            >>> shape = [N, C]
            >>> label_smoothing = 0.4
            >>> reduction='mean'
            >>> weight: Optional[paddle.Tensor]  = None
            >>> logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
            >>> integer_labels = paddle.randint(low=0, high=C, shape=[N], dtype='int64')
            >>> one_hot_labels = paddle.nn.functional.one_hot(integer_labels, C).astype('float32')

            >>> cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
            ...     weight=weight, reduction=reduction, label_smoothing=label_smoothing)

            >>> # integer labels
            >>> integer_label_dy_ret = cross_entropy_loss(logits, integer_labels)
            >>> print(integer_label_dy_ret)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            1.10520368)

            >>> # one_hot labels
            >>> one_hot_label_dy_ret = cross_entropy_loss(logits, one_hot_labels)
            >>> print(one_hot_label_dy_ret)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            1.10520368)

    """

    weight: Tensor | None
    ignore_index: int
    reduction: _ReduceMode
    soft_label: bool
    axis: int
    use_softmax: bool
    label_smoothing: float
    name: str | None

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: _ReduceMode = 'mean',
        soft_label: bool = False,
        axis: int = -1,
        use_softmax: bool = True,
        label_smoothing: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.soft_label = soft_label
        self.axis = axis
        self.use_softmax = use_softmax
        self.label_smoothing = label_smoothing
        self.name = name

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        ret = paddle.nn.functional.cross_entropy(
            input,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=self.soft_label,
            axis=self.axis,
            use_softmax=self.use_softmax,
            label_smoothing=self.label_smoothing,
            name=self.name,
        )

        return ret


class HSigmoidLoss(Layer):
    """
    Hierarchical Sigmoid Layer.

    The hierarchical sigmoid organizes the classes into a complete binary tree to reduce the computational complexity
    and speed up the model training, especially the training of language model.
    Each leaf node of the complete binary tree represents a class(word) and each non-leaf node acts as a binary classifier.
    For each class(word), there's a unique path from root to itself, hsigmoid calculate the cost for each non-leaf node on
    the path, and sum them to get a total cost.
    Comparing to softmax, the OP can reduce the computational complexity from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the number of classes or the size of word dict.

    The OP supports default tree and custom tree. For the default tree, you can refer to `Hierarchical Probabilistic Neural
    Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>_`. For the custom
    tree, you need to set :attr:`is_custom` to True, and do the following steps (take the language model as an example):

    1. Using a custom word dict to build a binary tree, each leaf node should be an word in the word dict.
    2. Creating a dict map word_id -> path that from the word to the root node, we call it path_table.
    3. Creating a dict map word_id -> code of path that from the word to the root node, we call it path_code.
       Code means the label of each binary classifier, 1 indicate true, 0 indicate false.
    4. Now, each word should has its path and code along the path, you can pass a batch of path and code related
       to the same batch of inputs.

    Parameters:
        feature_size (int): The number of features.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (:attr:`is_custom` is set to False), :attr:`num_classes`
            should not be None. If the custom tree is used (:attr:`is_custom` is set to True),
            :attr:`num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        weight_attr (ParamAttr|None, optional): The parameter attribute for the learnable weights
            of hsigmoid. If it is set to None or one attribute of ParamAttr, hsigmoid will create a
            ParamAttr as param_attr. If the Initializer of the param_attr is not set, the parameter is
            initialized with Xavier. Default is None.
        bias_attr (ParamAttr|bool|None, optional): The parameter attribute for the bias of hsigmoid. If it
            is set to False, no bias will be added. If it is set to None or one attribute of ParamAttr,
            hsigmoid will create a ParamAttr as bias_attr. If the Initializer of the bias_attr is not
            set, the bias is initialized zero. Default is None.
        is_custom (bool, optional): Whether use custom binary tree. If it's True, `path_table` and
            `path_code` should be passed to its forward method, otherwise `path_table` and `path_code`
            should not be passed to its forward method. Default is False.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating, if it's True,
            the gradient of weight and input will be sparse. Default is False.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        input (Tensor): The input tensor. The shapes is [N, D], where N is batch size and D is feature size. It's data type should be float32, float64.
        label (Tensor): It's shapes is [N, 1]. It's data type should be int64.
        output (Tensor): The HSigmoid Loss of ``input`` and ``label``. Shape is [N, 1]

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.set_device('cpu')
            >>> paddle.seed(2023)
            >>> input = paddle.uniform([4, 3])
            >>> print(input)
            Tensor(shape=[4, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0.73167229,  0.04029441, -0.48078126],
             [ 0.81050646, -0.15199822, -0.18717426],
             [ 0.94041789,  0.48874724,  0.03570259],
             [ 0.46585739,  0.95573163, -0.91368192]])
            >>> label = paddle.to_tensor([0, 1, 4, 5])
            >>> m = paddle.nn.HSigmoidLoss(3, 6)
            >>> out = m(input, label)
            >>> print(out)
            Tensor(shape=[4, 1], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[1.94512916],
             [2.26129627],
             [2.36135936],
             [2.97453213]])
    """

    weight: Tensor
    bias: Tensor

    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        weight_attr: ParamAttrLike | None = None,
        bias_attr: ParamAttrLike | None = None,
        is_custom: bool = False,
        is_sparse: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if (num_classes < 2) and (not is_custom):
            raise ValueError(
                "num_classes must not be less than 2 with default tree"
            )

        if (not is_custom) and (is_sparse):
            print("Sparse mode should not be used without custom tree")
            is_sparse = False

        self._feature_size = feature_size
        self._num_classes = num_classes
        self._is_custom = is_custom
        self._is_sparse = is_sparse

        self._weight_attr = weight_attr
        self._bias_attr = bias_attr

        self._name = name
        self._dtype = paddle.get_default_dtype()

        remote_prefetch = is_sparse
        print(
            "With sparse mode, if your models has only"
            " small parameter prefetch may cause speed down"
        )

        C = self._num_classes if is_custom else self._num_classes - 1
        self.weight = self.create_parameter(
            [C, self._feature_size],
            attr=self._weight_attr,
            is_bias=False,
            dtype=self._dtype,
        )
        self.bias = self.create_parameter(
            [C, 1], attr=self._bias_attr, is_bias=True, dtype=self._dtype
        )

    def forward(
        self,
        input: Tensor,
        label: Tensor,
        path_table: Tensor = None,
        path_code: Tensor = None,
    ) -> Tensor:
        out = F.hsigmoid_loss(
            input,
            label,
            self._num_classes,
            self.weight,
            self.bias,
            path_table=path_table,
            path_code=path_code,
            is_sparse=self._is_sparse,
            name=self._name,
        )
        return out


class MSELoss(Layer):
    r"""
    **Mean Square Error Loss**
    Computes the mean square error (squared L2 norm) of given input and label.

    If :attr:`reduction` is set to ``'none'``, loss is calculated as:

    .. math::
        Out = (input - label)^2

    If :attr:`reduction` is set to ``'mean'``, loss is calculated as:

    .. math::
        Out = \operatorname{mean}((input - label)^2)

    If :attr:`reduction` is set to ``'sum'``, loss is calculated as:

    .. math::
        Out = \operatorname{sum}((input - label)^2)

    where `input` and `label` are `float32` tensors of same shape.

    Parameters:
        reduction (str, optional): The reduction method for the output,
            could be 'none' | 'mean' | 'sum'.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned.
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.

    Shape:
        input (Tensor): Input tensor, the data type is float32 or float64
        label (Tensor): Label tensor, the data type is float32 or float64
        output (Tensor): output tensor storing the MSE loss of input and label, the data type is same as input.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> mse_loss = paddle.nn.loss.MSELoss()
            >>> input = paddle.to_tensor([1.5])
            >>> label = paddle.to_tensor([1.7])
            >>> output = mse_loss(input, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.04000002)

    """

    reduction: _ReduceMode

    def __init__(self, reduction: _ReduceMode = 'mean'):
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "'reduction' in 'MSELoss' should be 'sum', 'mean' or 'none', "
                f"but received {reduction}."
            )
        self.reduction = reduction

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        if not in_dynamic_mode():
            base.data_feeder.check_variable_and_dtype(
                input, 'input', ['float32', 'float64'], 'MSELoss'
            )
            base.data_feeder.check_variable_and_dtype(
                label, 'label', ['float32', 'float64'], 'MSELoss'
            )

        if in_dynamic_or_pir_mode():
            square_out = paddle._C_ops.square(paddle.subtract(input, label))
        else:
            square_out = paddle.square(paddle.subtract(input, label))
        if self.reduction == 'none':
            return square_out

        reduce_op = 'reduce_mean'
        if self.reduction == 'sum':
            square_out = paddle.sum(square_out)
            return square_out

        return paddle.mean(square_out)


class L1Loss(Layer):
    r"""

    Construct a callable object of the ``L1Loss`` class.
    The L1Loss layer calculates the L1 Loss of ``input`` and ``label`` as follows.

    If `reduction` set to ``'none'``, the loss is:

    .. math::
        Out = \lvert input - label\rvert

    If `reduction` set to ``'mean'``, the loss is:

    .. math::
        Out = MEAN(\lvert input - label\rvert)

    If `reduction` set to ``'sum'``, the loss is:

    .. math::
        Out = SUM(\lvert input - label\rvert)


    Parameters:
        reduction (str, optional): Indicate the reduction to apply to the loss,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'none'``, the unreduced loss is returned;
            If `reduction` is ``'mean'``, the reduced mean loss is returned.
            If `reduction` is ``'sum'``, the reduced sum loss is returned.
            Default is ``'mean'``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input (Tensor): The input tensor. The shapes is ``[N, *]``, where N is batch size and `*` means any number of additional dimensions. It's data type should be float32, float64, int32, int64.
        - label (Tensor): label. The shapes is ``[N, *]``, same shape as ``input`` . It's data type should be float32, float64, int32, int64.
        - output (Tensor): The L1 Loss of ``input`` and ``label``.
          If `reduction` is ``'none'``, the shape of output loss is ``[N, *]``, the same as ``input`` .
          If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [].

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([[1.5, 0.8], [0.2, 1.3]])
            >>> label = paddle.to_tensor([[1.7, 1], [0.4, 0.5]])

            >>> l1_loss = paddle.nn.L1Loss()
            >>> output = l1_loss(input, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.34999999)

            >>> l1_loss = paddle.nn.L1Loss(reduction='sum')
            >>> output = l1_loss(input, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.39999998)

            >>> l1_loss = paddle.nn.L1Loss(reduction='none')
            >>> output = l1_loss(input, label)
            >>> print(output)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.20000005, 0.19999999],
             [0.20000000, 0.79999995]])

    """

    reduction: _ReduceMode
    name: str | None

    def __init__(
        self, reduction: _ReduceMode = 'mean', name: str | None = None
    ) -> None:
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in L1Loss should be 'sum', 'mean' or 'none', but "
                f"received {reduction}, which is not allowed."
            )
        super().__init__()
        self.reduction = reduction
        self.name = name

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        return paddle.nn.functional.l1_loss(
            input, label, self.reduction, name=self.name
        )


class BCELoss(Layer):
    """

    This interface is used to construct a callable object of the ``BCELoss`` class.
    The BCELoss layer measures the binary_cross_entropy loss between input predictions ``input``
    and target labels ``label`` . The binary_cross_entropy loss can be described as:

    If :attr:`weight` is set, the loss is:

    .. math::
        Out = -1 * weight * (label * log(input) + (1 - label) * log(1 - input))

    If :attr:`weight` is None, the loss is:

    .. math::
        Out = -1 * (label * log(input) + (1 - label) * log(1 - input))

    If :attr:`reduction` set to ``'none'``, the interface will return the original loss `Out`.

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(Out)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(Out)

    Note that the input predictions ``input`` always be the output of sigmoid, and the target labels ``label``
    should be numbers between 0 and 1.

    Parameters:
        weight (Tensor, optional): A manual rescaling weight given to the loss of each
            batch element. If given, has to be a Tensor of size nbatch and the data type
            is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input (Tensor): 2-D tensor with shape: ``[N, *]``, N is batch_size, `*` means number of additional dimensions. The input ``input`` should always be the output of sigmod. Available dtype is float16, float32, float64.
        - label (Tensor): 2-D tensor with the same shape as ``input``. The target labels which values should be numbers between 0 and 1. Available dtype is float16, float32, float64.
        - output (Tensor): If ``reduction`` is ``'none'``, the shape of output is same as ``input`` , else the shape of output is scalar.

    Returns:
        A callable object of BCELoss.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([0.5, 0.6, 0.7])
            >>> label = paddle.to_tensor([1.0, 0.0, 1.0])
            >>> bce_loss = paddle.nn.BCELoss()
            >>> output = bce_loss(input, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.65537095)

    """

    weight: Tensor | None
    reduction: _ReduceMode
    name: str | None

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: _ReduceMode = 'mean',
        name: str | None = None,
    ):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in bce_loss should be 'sum', 'mean' or 'none', but "
                f"received {reduction}, which is not allowed."
            )

        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.name = name

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        out = paddle.nn.functional.binary_cross_entropy(
            input, label, self.weight, self.reduction, self.name
        )
        return out


class NLLLoss(Layer):
    r"""

    This class accepts input and target label and returns negative log likelihood
    cross error. It is useful to train a classification problem with C classes.

    The input for the loss is expected to contain log-probabilities of
    each classes. It has to be a Tensor of size either (batch_size, C) or
    (batch_size, C, d1, d2, ..., dK) with K >= 1 for the K-dimensional case.
    The label for the loss should be a class index in the range [0, C-1]
    where C is the number of classes. If ignore_index is specified, the
    specified target value does not contribute to the input gradient.

    If the optional argument `weight` is provided, it should be a 1D Tensor
    assigning weight to each of the classed. This is particularly useful
    when you have an unbalanced training set.

    The loss is calculated as follows.
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::

        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore_index}\},

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::

        \ell(x, y) =
        \left\{
            \begin{array}{lcl}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if  reduction} = \text{'mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if  reduction} = \text{'sum'.}
            \end{array}
        \right.

    Parameters:
        weight (Tensor, optional): Weight tensor, a manual rescaling weight given
            to each class. If given, it has to be a 1D Tensor whose size is `[C, ]`. Otherwise,
            it treated as if having all ones. the data type is
            float32, float64, Default is ``'None'``.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient.
        reduction (str, optional): Indicate how to average the loss,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``. Default is ``'mean'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            if `reduction` is ``'sum'``, the reduced sum loss is returned;
            if `reduction` is ``'none'``, no reduction will be applied.
            Default is ``'mean'``.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default is ``'None'``.

    Shape:
        - input (Tensor): Input tensor, the shape is :math:`[N, C]`, `C` is the number of classes.
            But in K-dimension situation, the shape is :math:`[N, C, d_1, d_2, ..., d_K]`.
            The data type is float32, float64.
        - label (Tensor): Label tensor, the shape is :math:`[N,]` or :math:`[N, d_1, d_2, ..., d_K]`.
            The data type is int64.
        - output (Tensor): the `negative log likelihood loss` between input `x` and `label`.
            If `reduction` is `'none'`, the shape is `[N, *]`.
            If `reduction` is `'sum'` or `'mean'`, the shape is `[]`.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> nll_loss = paddle.nn.loss.NLLLoss()
            >>> log_softmax = paddle.nn.LogSoftmax(axis=1)

            >>> input = paddle.to_tensor([[0.88103855, 0.9908683 , 0.6226845 ],
            ...                           [0.53331435, 0.07999352, 0.8549948 ],
            ...                           [0.25879037, 0.39530203, 0.698465  ],
            ...                           [0.73427284, 0.63575995, 0.18827209],
            ...                           [0.05689114, 0.0862954 , 0.6325046 ]], "float32")
            >>> log_out = log_softmax(input)
            >>> label = paddle.to_tensor([0, 2, 1, 1, 0], "int64")
            >>> result = nll_loss(log_out, label)
            >>> print(result)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.07202101)

    """

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: _ReduceMode = 'mean',
        name: str | None = None,
    ) -> None:
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in nll_loss should be 'sum', 'mean' or "
                f"'none', but received {reduction}, which is not allowed."
            )
        super().__init__()
        self._weight = weight
        self._ignore_index = ignore_index
        self._reduction = reduction
        self._name = name

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        return F.nll_loss(
            input,
            label,
            weight=self._weight,
            ignore_index=self._ignore_index,
            reduction=self._reduction,
            name=self._name,
        )


class PoissonNLLLoss(Layer):
    r"""Generate a callable object of 'PoissonNLLLoss' to calculate the
    Poisson negative log likelihood loss between Input(input) and
    Input(label). Notes that Input(input) is the expectation of underlying
    Poisson distribution and Input(label) is the random samples from the
    Poisson distribution


    Poisson negative log likelihood loss is calculated as follows:

    .. math::
        \text{loss}(\text{input}, \text{label}) = \text{input} - \text{label} * \log(\text{label}) + \log(\text{label!})

    The last term can be approximated with Stirling formula. This approximation term is used when :attr:`full` is ``True``.
    The approximation is added when label values are more than 1 and omitted when the labels are less than or equal to 1.

    Parameters:
         log_input (bool, optional):
            Whether to the treat input tensor as log input.
            If ``True`` the loss is computed as, :math:`\exp(\text{input}) - \text{label} * \text{input}` .
            If ``False`` then loss is :math:`\text{input} - \text{label} * \log(\text{input}+\text{epsilon})` .
            Default: ``True``.
         full (bool, optional):
            Whether to compute full loss.
            If ``True``, the Stirling approximation term is added.
            If ``False``, the Stirling approximation is dropped.
            Default: ``False``.
         epsilon (float, optional):
            A small value to avoid evaluation of :math:`\log(0)` when ``log_input`` = ``False``. ``epsilon > 0``.
            Default: 1e-8.
         reduction (str, optional):
            Indicate how to reduce the loss, the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            if `reduction` is ``'sum'``, the reduced sum loss is returned;
            if `reduction` is ``'none'``, no reduction will be applied.
            Default is ``'mean'``.
         name (str|None, optional):
            Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input (Tensor): The shape of input tensor should be `(N, *)` or `(*)` where `(*)` denotes any number of extra dimensions.
        - label (Tensor): The shape of input tensor should be `(N, *)` or `(*)`, same shape as the input tensor.
        - output (Tensor): scalar if :attr:`reduction` is ``'mean'`` (default) or ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same shape as the input

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> poisson_nll_loss = paddle.nn.loss.PoissonNLLLoss()
            >>> input = paddle.randn([5, 2], dtype=paddle.float32)
            >>> label = paddle.randn([5, 2], dtype=paddle.float32)
            >>> loss = poisson_nll_loss(input, label)
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.52983975)

    """

    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        epsilon: float = 1e-08,
        reduction: _ReduceMode = 'mean',
        name: str | None = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(
                f"The value of `epsilon` in PoissonNLLLoss should be positive, but received {epsilon:f}, which is not allowed"
            )
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in PoissonNLLLoss should be 'sum', 'mean' or 'none', but "
                f"received {reduction}, which is not allowed."
            )
        super().__init__()
        self._log_input = log_input
        self._full = full
        self._epsilon = epsilon
        self._reduction = reduction
        self._name = name

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        return F.poisson_nll_loss(
            input,
            label,
            log_input=self._log_input,
            full=self._full,
            epsilon=self._epsilon,
            reduction=self._reduction,
            name=self._name,
        )


class KLDivLoss(Layer):
    r"""

    Generate a callable object of 'KLDivLoss' to calculate the
    Kullback-Leibler divergence loss between Input(X) and
    Input(Target). Notes that Input(X) is the log-probability
    and Input(Target) is the probability.

    KL divergence loss is calculated as follows:

    If `log_target` is False:

    $$l(x, y) = y * (\log(y) - x)$$

    If `log_target` is True:

    $$l(x, y) = \exp(y) * (y - x)$$

    Here :math:`x` is input and :math:`y` is label.

    If `reduction` is ``'none'``, the output loss is the same shape as the input, and the loss at each point is calculated separately. There is no reduction to the result.

    If `reduction` is ``'mean'``, the output loss is the shape of [], and the output is the average of all losses.

    If `reduction` is ``'sum'``, the output loss is the shape of [], and the output is the sum of all losses.

    If `reduction` is ``'batchmean'``, the output loss is the shape of [N], N is the batch size, and the output is the sum of all losses divided by the batch size.

    Parameters:
        reduction (str, optional): Indicate how to average the loss,
            the candidates are ``'none'`` | ``'batchmean'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
            if `reduction` is ``'sum'``, the reduced sum loss is returned;
            if `reduction` is ``'none'``, no reduction will be applied.
            Default is ``'mean'``.
        log_target (bool, optional): Indicate whether `label` is passed in log space. Default is False.

    Shape:

        input (Tensor): ``(N, *)``, where ``*`` means, any number of additional dimensions.

        label (Tensor): ``(N, *)``, same shape as input.

        output (Tensor): tensor with shape: [] by default.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> shape = (5, 20)
            >>> x = paddle.uniform(shape, min=-10, max=10).astype('float32')
            >>> target = paddle.uniform(shape, min=-10, max=10).astype('float32')

            >>> # 'batchmean' reduction, loss shape will be []
            >>> kldiv_criterion = nn.KLDivLoss(reduction='batchmean')
            >>> pred_loss = kldiv_criterion(x, target)
            >>> print(pred_loss.shape)
            []

            >>> # 'mean' reduction, loss shape will be []
            >>> kldiv_criterion = nn.KLDivLoss(reduction='mean')
            >>> pred_loss = kldiv_criterion(x, target)
            >>> print(pred_loss.shape)
            []

            >>> # 'sum' reduction, loss shape will be []
            >>> kldiv_criterion = nn.KLDivLoss(reduction='sum')
            >>> pred_loss = kldiv_criterion(x, target)
            >>> print(pred_loss.shape)
            []

            >>> # 'none' reduction, loss shape is same with X shape
            >>> kldiv_criterion = nn.KLDivLoss(reduction='none')
            >>> pred_loss = kldiv_criterion(x, target)
            >>> print(pred_loss.shape)
            [5, 20]

            >>> # if label is in the log space, set log_target = True
            >>> target = paddle.uniform(shape, min=0, max=10).astype('float32')
            >>> log_target = paddle.log(target)
            >>> kldiv_criterion_1 = nn.KLDivLoss(reduction='none')
            >>> kldiv_criterion_2 = nn.KLDivLoss(reduction='none', log_target=True)
            >>> pred_loss_1 = kldiv_criterion_1(x, target)
            >>> pred_loss_2 = kldiv_criterion_2(x, log_target)
            >>> print(paddle.allclose(pred_loss_1, pred_loss_2))
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            True)
    """

    reduction: _ReduceMode
    log_target: bool

    def __init__(
        self, reduction: _ReduceMode = 'mean', log_target: bool = False
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        out = F.kl_div(input, label, self.reduction, self.log_target)
        return out


class MarginRankingLoss(Layer):
    r"""

    This interface is used to construct a callable object of the ``MarginRankingLoss`` class.
    The MarginRankingLoss layer calculates the margin rank loss between the input, other and label
    , use the math function as follows.

    .. math::
        margin\_rank\_loss = max(0, -label * (input - other) + margin)

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(margin\_rank\_loss)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(margin\_rank\_loss)

    If :attr:`reduction` set to ``'none'``, just return the origin ``margin_rank_loss``.

    Parameters:
        margin (float, optional): The margin value to add, default value is 0;
        reduction (str, optional): Indicate the reduction to apply to the loss, the candidates are ``'none'``, ``'mean'``, ``'sum'``.If :attr:`reduction` is ``'none'``, the unreduced loss is returned; If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned. If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned. Default is ``'mean'``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:

        input: N-D Tensor, the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.

        other: N-D Tensor, `other` have the same shape and dtype as `input`.

        label: N-D Tensor, label have the same shape and dtype as `input`.

        output: If :attr:`reduction` is ``'mean'`` or ``'sum'`` , the out shape is :math:`[]`, otherwise the shape is the same as `input` .The same dtype as input tensor.

    Returns:
        A callable object of MarginRankingLoss.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([[1, 2], [3, 4]], dtype="float32")
            >>> other = paddle.to_tensor([[2, 1], [2, 4]], dtype="float32")
            >>> label = paddle.to_tensor([[1, -1], [-1, -1]], dtype="float32")
            >>> margin_rank_loss = paddle.nn.MarginRankingLoss()
            >>> loss = margin_rank_loss(input, other, label)
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.75000000)
    """

    margin: float
    reduction: _ReduceMode
    name: str | None

    def __init__(
        self,
        margin: float = 0.0,
        reduction: _ReduceMode = 'mean',
        name: str | None = None,
    ) -> None:
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in MarginRankingLoss should be 'sum', 'mean' or 'none', but "
                f"received {reduction}, which is not allowed."
            )
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.name = name

    def forward(self, input: Tensor, other: Tensor, label: Tensor) -> Tensor:
        out = paddle.nn.functional.margin_ranking_loss(
            input, other, label, self.margin, self.reduction, self.name
        )
        return out


class CTCLoss(Layer):
    r"""

    An operator integrating the open source Warp-CTC library (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation
    is integrated to the Warp-CTC library to normalize values for each row of the input tensor.

    Parameters:
        blank (int, optional): The blank label index of Connectionist Temporal Classification (CTC) loss, which is in the half-opened interval [0, num_classes + 1). The data type must be int32. Default is 0.
        reduction (string, optional): Indicate how to average the loss, the candidates are ``'none'`` | ``'mean'`` | ``'sum'``. If :attr:`reduction` is ``'mean'``, the output loss will be divided by the label_lengths, and then return the mean of quotient; If :attr:`reduction` is ``'sum'``, return the sum of loss; If :attr:`reduction` is ``'none'``, no reduction will be applied. Default is ``'mean'``.

    Shape:
        - log_probs (Tensor): The unscaled probability sequence with padding, which is a 3-D Tensor. The tensor shape is [max_logit_length, batch_size, num_classes + 1], where max_logit_length is the longest length of input logit sequence. The data type should be float32 or float64.
        - labels (Tensor): The ground truth sequence with padding, which must be a 3-D Tensor. The tensor shape is [batch_size, max_label_length], where max_label_length is the longest length of label sequence. The data type must be int32.
        - input_lengths (Tensor): The length for each input sequence, it should have shape [batch_size] and dtype int64.
        - label_lengths (Tensor): The length for each label sequence, it should have shape [batch_size] and dtype int64.
        - norm_by_times (bool, optional): Whether to normalize the gradients by the number of time-step, which is also the sequence's length. There is no need to normalize the gradients if reduction mode is 'mean'. Default: False.

    Returns:
        Tensor, The Connectionist Temporal Classification (CTC) loss between ``log_probs`` and  ``labels``. If attr:`reduction` is ``'none'``, the shape of loss is [batch_size], otherwise, the shape of loss is []. Data type is the same as ``log_probs``.

    Examples:

        .. code-block:: python

            >>> # declarative mode
            >>> import paddle

            >>> # length of the longest logit sequence
            >>> max_seq_length = 4
            >>> #length of the longest label sequence
            >>> max_label_length = 3
            >>> # number of logit sequences
            >>> batch_size = 2
            >>> # class num
            >>> class_num = 3

            >>> log_probs = paddle.to_tensor([[[4.17021990e-01, 7.20324516e-01, 1.14374816e-04],
            ...                                [3.02332580e-01, 1.46755889e-01, 9.23385918e-02]],
            ...                               [[1.86260208e-01, 3.45560730e-01, 3.96767467e-01],
            ...                                [5.38816750e-01, 4.19194520e-01, 6.85219526e-01]],
            ...                               [[2.04452246e-01, 8.78117442e-01, 2.73875929e-02],
            ...                                [6.70467496e-01, 4.17304814e-01, 5.58689833e-01]],
            ...                               [[1.40386939e-01, 1.98101491e-01, 8.00744593e-01],
            ...                                [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]],
            ...                               [[8.76389146e-01, 8.94606650e-01, 8.50442126e-02],
            ...                                [3.90547849e-02, 1.69830427e-01, 8.78142476e-01]]], dtype="float32")
            >>> labels = paddle.to_tensor([[1, 2, 2], [1, 2, 2]], dtype="int32")
            >>> input_lengths = paddle.to_tensor([5, 5], dtype="int64")
            >>> label_lengths = paddle.to_tensor([3, 3], dtype="int64")

            >>> loss = paddle.nn.CTCLoss(blank=0, reduction='none')(log_probs, labels, input_lengths, label_lengths)
            >>> print(loss)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [3.91798496, 2.90765214])

            >>> loss = paddle.nn.CTCLoss(blank=0, reduction='mean')(log_probs, labels, input_lengths, label_lengths)
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.13760614)
    """

    blank: int
    reduction: _ReduceMode

    def __init__(self, blank: int = 0, reduction: _ReduceMode = 'mean') -> None:
        super().__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(
        self,
        log_probs: Tensor,
        labels: Tensor,
        input_lengths: Tensor,
        label_lengths: Tensor,
        norm_by_times: bool = False,
    ) -> Tensor:
        return paddle.nn.functional.ctc_loss(
            log_probs,
            labels,
            input_lengths,
            label_lengths,
            self.blank,
            self.reduction,
            norm_by_times=norm_by_times,
        )


class RNNTLoss(Layer):
    """
    Parameters:
        blank (int, optional): blank label. Default: 0.
        fastemit_lambda (float, optional): Regularization parameter for FastEmit (https://arxiv.org/pdf/2010.11148.pdf)
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: 'mean'

    Shape:
        input: logprob Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        label: 2 dimensional (batch, labelLength) Tensor containing all the targets of the batch with zero padded
        input_lengths: Tensor of size (batch) containing size of each output sequence from the network
        label_lengths: Tensor of (batch) containing label length of each example

    Returns:
        Tensor, The RNN-T loss between ``logprobs`` and  ``labels``. If attr:`reduction` is ``'none'``, the shape of loss is [batch_size], otherwise, the shape of loss is []. Data type is the same as ``logprobs``.

    Examples:
        .. code-block:: python

            >>> # declarative mode
            >>> import numpy as np
            >>> import paddle
            >>> from paddle.nn import RNNTLoss

            >>> fn = RNNTLoss(reduction='sum', fastemit_lambda=0.0)

            >>> acts = np.array([[[[0.1, 0.6, 0.1, 0.1, 0.1],
            ...                    [0.1, 0.1, 0.6, 0.1, 0.1],
            ...                    [0.1, 0.1, 0.2, 0.8, 0.1]],
            ...                   [[0.1, 0.6, 0.1, 0.1, 0.1],
            ...                    [0.1, 0.1, 0.2, 0.1, 0.1],
            ...                    [0.7, 0.1, 0.2, 0.1, 0.1]]]])
            >>> labels = [[1, 2]]

            >>> acts = paddle.to_tensor(acts, stop_gradient=False)

            >>> lengths = [acts.shape[1]] * acts.shape[0]
            >>> label_lengths = [len(l) for l in labels]
            >>> labels = paddle.to_tensor(labels, paddle.int32)
            >>> lengths = paddle.to_tensor(lengths, paddle.int32)
            >>> label_lengths = paddle.to_tensor(label_lengths, paddle.int32)

            >>> costs = fn(acts, labels, lengths, label_lengths)
            >>> print(costs)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
            -2.85042444)
    """

    blank: int
    fastemit_lambda: float
    reduction: _ReduceMode
    name: str | None

    def __init__(
        self,
        blank: int = 0,
        fastemit_lambda: float = 0.001,
        reduction: _ReduceMode = 'mean',
        name=None,
    ) -> None:
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.fastemit_lambda = fastemit_lambda
        self.name = name

    def forward(
        self,
        input: Tensor,
        label: Tensor,
        input_lengths: Tensor,
        label_lengths: Tensor,
    ) -> Tensor:
        return paddle.nn.functional.rnnt_loss(
            input,
            label,
            input_lengths,
            label_lengths,
            blank=self.blank,
            fastemit_lambda=self.fastemit_lambda,
            reduction=self.reduction,
            name=self.name,
        )


class SmoothL1Loss(Layer):
    r"""
    This operator calculates smooth_l1_loss. Creates a criterion that uses a squared
    term if the absolute element-wise error falls below 1 and an L1 term otherwise.
    In some cases it can prevent exploding gradients and it is more robust and less
    sensitivity to outliers. Also known as the Huber loss:

    .. math::

        loss(x, y) = \frac{1}{n}\sum_{i}z_i

    where :math:`z_i` is given by:

    .. math::

        \mathop{z_i} = \left\{\begin{array}{rcl}
                0.5(x_i - y_i)^2 & & {if |x_i - y_i| < \delta} \\
                \delta * |x_i - y_i| - 0.5 * \delta^2 & & {otherwise}
            \end{array} \right.

    Parameters:
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        delta (float, optional): Specifies the hyperparameter :math:`\delta` to be used.
            The value determines how large the errors need to be to use L1. Errors
            smaller than delta are minimized with L2. Parameter is ignored for
            negative/zero values. Default value is :math:`1.0`.
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Call Parameters:

        input (Tensor): Input tensor, the data type is float32 or float64. Shape is (N, C),
        where C is number of classes, and if shape is more than 2D,
        this is (N, C, D1, D2,..., Dk), k >= 1.

        label (Tensor): Label tensor, the data type is float32 or float64.
        The shape of label is the same as the shape of input.

    Returns:
        Tensor, The tensor storing the smooth_l1_loss of input and label.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> input = paddle.rand([3, 3]).astype("float32")
            >>> label = paddle.rand([3, 3]).astype("float32")
            >>> loss = paddle.nn.SmoothL1Loss()
            >>> output = loss(input, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.08307374)
    """

    reduction: _ReduceMode
    delta: float
    name: str | None

    def __init__(
        self,
        reduction: _ReduceMode = 'mean',
        delta: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.delta = delta
        self.name = name

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        return F.smooth_l1_loss(
            input,
            label,
            reduction=self.reduction,
            delta=self.delta,
            name=self.name,
        )


class MultiLabelSoftMarginLoss(Layer):
    r"""Creates a criterion that optimizes a multi-class multi-classification
    hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
    and output :math:`y` (which is a 2D `Tensor` of target class indices).
    For each sample in the mini-batch:

    .. math::
        \text{loss}(x, y) = - \frac{1}{C} * \sum_i y[i] * \log((1 + \exp(-x[i]))^{-1})
            + (1-y[i]) * \log\left(\frac{\exp(-x[i])}{(1 + \exp(-x[i]))}\right)

    where :math:`i \in \left\{0, \; \cdots , \; \text{x.nElement}() - 1\right\}`,
    :math:`y[i] \in \left\{0, \; 1\right\}`.

    Parameters:
        weight (Tensor,optional): a manual rescaling weight given to each class.
                If given, has to be a Tensor of size C and the data type is float32, float64.
                Default is ``'None'`` .
        reduction (str, optional): Indicate how to average the loss by batch_size,
                the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
                If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
                If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
                If :attr:`reduction` is ``'sum'``, the summed loss is returned.
                Default: ``'mean'``
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Call parameters:
        input (Tensor): Input tensor, the data type is float32 or float64. Shape is (N, C), where C is number of classes, and if shape is more than 2D, this is (N, C, D1, D2,..., Dk), k >= 1.
        label (Tensor): Label tensor containing 1 or -1, the data type is float32 or float64. The shape of label is the same as the shape of input.

    Shape:
        input: N-D Tensor, the shape is [N, \*], N is batch size and `\*` means number of classes, available dtype is float32, float64. The sum operationoperates over all the elements.
        label: N-D Tensor, same shape as the input.
        output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input.

    Returns:
        A callable object of MultiLabelSoftMarginLoss.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
            >>> label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)

            >>> multi_label_soft_margin_loss = nn.MultiLabelSoftMarginLoss(reduction='none')
            >>> loss = multi_label_soft_margin_loss(input, label)
            >>> print(loss)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [3.49625897, 0.71111226, 0.43989015])

            >>> multi_label_soft_margin_loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
            >>> loss = multi_label_soft_margin_loss(input, label)
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.54908717)
    """

    weight: Tensor | None
    reduction: _ReduceMode
    name: str | None

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: _ReduceMode = 'mean',
        name: str | None = None,
    ) -> None:
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "'reduction' in 'MultiLabelSoftMarginloss' should be 'sum', 'mean' or 'none', "
                f"but received {reduction}."
            )
        self.weight = weight
        self.reduction = reduction
        self.name = name

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        return F.multi_label_soft_margin_loss(
            input,
            label,
            weight=self.weight,
            reduction=self.reduction,
            name=self.name,
        )


class HingeEmbeddingLoss(Layer):
    r"""
    Create a callable object of `HingeEmbeddingLoss` to calculates hinge_embedding_loss. Measures the loss given an input tensor :math:`x` and a labels tensor :math:`y`(containing 1 or -1).
    This is usually used for measuring whether two inputs are similar or dissimilar, e.g. using the L1 pairwise distance as :math:`x`,
    and is typically used for learning nonlinear embeddings or semi-supervised learning.

    The loss function for :math:`n`-th sample in the mini-batch is

    .. math::
        l_n = \begin{cases}
            x_n, & \text{if}\; y_n = 1,\\
            \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Parameters:

        margin (float, optional): Specifies the hyperparameter margin to be used.
            The value determines how large the input need to be to calculate in
            hinge_embedding_loss. When label is -1, Input smaller than margin are minimized with hinge_embedding_loss.
            Default = 1.0
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default: ``'mean'``
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Call Parameters:

        input (Tensor): Input tensor, the data type is float32 or float64. Shape is (N, C), where C is number of classes, and if shape is more than 2D, this is (N, C, D1, D2,..., Dk), k >= 1.

        label (Tensor): Label tensor containing 1 or -1, the data type is float32 or float64. The shape of label is the same as the shape of input.

    Shape:

        input: N-D Tensor, the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64. The sum operationoperates over all the elements.

        label: N-D Tensor, same shape as the input.

        output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input.

    Returns:

        Tensor, The tensor variable storing the hinge_embedding_loss of input and label.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
            >>> # label elements in {1., -1.}
            >>> label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)

            >>> hinge_embedding_loss = nn.HingeEmbeddingLoss(margin=1.0, reduction='none')
            >>> loss = hinge_embedding_loss(input, label)
            >>> print(loss)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[ 0., -2.,  0.],
             [ 0., -1.,  2.],
             [ 1.,  1.,  1.]])

            >>> hinge_embedding_loss = nn.HingeEmbeddingLoss(margin=1.0, reduction='mean')
            >>> loss = hinge_embedding_loss(input, label)
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.22222222)
    """

    margin: float
    reduction: _ReduceMode
    name: str | None

    def __init__(
        self,
        margin: float = 1.0,
        reduction: _ReduceMode = 'mean',
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.name = name

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        return F.hinge_embedding_loss(
            input,
            label,
            reduction=self.reduction,
            margin=self.margin,
            name=self.name,
        )


class CosineEmbeddingLoss(Layer):
    r"""
    This interface is used to construct a callable object of the ``CosineEmbeddingLoss`` class.
    The CosineEmbeddingLoss layer measures the cosine_embedding loss between input predictions ``input1``, ``input2``
    and target labels ``label`` with values 1 or 0. This is used for measuring whether two inputs are similar or
    dissimilar and is typically used for learning nonlinear embeddings or semi-supervised learning.
    The cosine embedding loss can be described as:

    If label = 1, then the loss value can be calculated as follow:

    .. math::
        Out = 1 - cos(input1, input2)

    If label = -1, then the loss value can be calculated as follow:

    .. math::
        Out = max(0, cos(input1, input2)) - margin

    The operator cos can be described as follow:
     .. math::
        cos(x1, x2) = \frac{x1 \cdot{} x2}{\Vert x1 \Vert_2 * \Vert x2 \Vert_2}

    Parameters:
        margin (float, optional): Should be a number from :math:`-1` to :math:`1`,
            :math:`0` to :math:`0.5` is suggested. If :attr:`margin` is missing, the
            default value is :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        input1 (Tensor): tensor with shape: [N, M] or [M], 'N' means batch size, which can be 0, 'M' means the length of input array.
                         Available dtypes are float32, float64.
        input2 (Tensor): tensor with shape: [N, M] or [M], 'N' means batch size, which can be 0, 'M' means the length of input array.
                         Available dtypes are float32, float64.
        label (Tensor): tensor with shape: [N] or [1], 'N' means the length of input array. The target labels values should be -1 or 1.
                         Available dtypes are int32, int64, float32, float64.
        output (Tensor): Tensor, the cosine embedding Loss of Tensor ``input1`` ``input2`` and ``label``.
                         If `reduction` is ``'none'``, the shape of output loss is [N], the same as ``input`` .
                         If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [].

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input1 = paddle.to_tensor([[1.6, 1.2, -0.5], [3.2, 2.6, -5.8]], 'float32')
            >>> input2 = paddle.to_tensor([[0.5, 0.5, -1.8], [2.3, -1.4, 1.1]], 'float32')
            >>> label = paddle.to_tensor([1, -1], 'int64')

            >>> cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='mean')
            >>> output = cosine_embedding_loss(input1, input2, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.21155193)

            >>> cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='sum')
            >>> output = cosine_embedding_loss(input1, input2, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.42310387)

            >>> cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='none')
            >>> output = cosine_embedding_loss(input1, input2, label)
            >>> print(output)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.42310387, 0.        ])

    """

    margin: float
    reduction: _ReduceMode
    name: str | None

    def __init__(
        self,
        margin: float = 0,
        reduction: _ReduceMode = 'mean',
        name: str | None = None,
    ) -> None:
        if margin > 1 or margin < -1:
            raise ValueError(
                f"The value of 'margin' should be in the interval of [-1, 1], but received {margin:f}, which is not allowed."
            )
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' should be 'sum', 'mean' or "
                f"'none', but received {reduction}, which is not allowed."
            )
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.name = name

    def forward(self, input1: Tensor, input2: Tensor, label: Tensor) -> Tensor:
        return F.cosine_embedding_loss(
            input1,
            input2,
            label,
            margin=self.margin,
            reduction=self.reduction,
            name=self.name,
        )


class TripletMarginWithDistanceLoss(Layer):
    r"""
    Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `input`, `positive` and `negative` (i.e., `input`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(input, pos, neg) = \max \{d(input_i, pos_i) - d(input_i, neg_i) + {\rm margin}, 0\}

    where the default `distance_function`

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_2

    or user can define their own distance function. `margin` is a nonnegative margin representing the minimum difference
    between the positive and negative distances that is required for the loss to be 0. If `swap` is true, it will compare distance of (input, negative) with
    distance of (negative, positive) and change it to the smaller one. For more details see http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf.

    Parameters:
        distance_function (Callable, Optional): Quantifies the distance between two tensors. if not specified, 2 norm functions will be used.

        margin (float, Optional):Default: :math:`1`.A nonnegative margin representing the minimum difference
                between the positive and negative distances required for the loss to be 0. Larger
                margins penalize cases where the negative examples are not distant enough from the
                anchors, relative to the positives.

        swap (bool, Optional):The distance swap changes the negative distance to the swap distance (distance between positive samples
                and negative samples) if swap distance smaller than negative distance. Default: ``False``.

        reduction (str, Optional):Indicate how to average the loss by batch_size.
                the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
                If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
                If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
                If :attr:`reduction` is ``'sum'``, the summed loss is returned.
                Default: ``'mean'``
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shapes:
      - input (Tensor):Input tensor, the data type is float32 or float64.
        the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.

      - positive (Tensor):Positive tensor, the data type is float32 or float64.
        The shape of label is the same as the shape of input.

      - negative (Tensor):Negative tensor, the data type is float32 or float64.
        The shape of label is the same as the shape of input.

      - output(Tensor): The tensor variable storing the triplet_margin_with_distance_loss of input and positive and negative.

    Return:
        A callable object of TripletMarginWithDistanceLoss

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.nn import TripletMarginWithDistanceLoss

            >>> input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
            >>> positive= paddle.to_tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]], dtype=paddle.float32)
            >>> negative = paddle.to_tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]], dtype=paddle.float32)
            >>> triplet_margin_with_distance_loss = TripletMarginWithDistanceLoss(reduction='none')
            >>> loss = triplet_margin_with_distance_loss(input, positive, negative,)
            >>> print(loss)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 0.57496595, 0.        ])

            >>> triplet_margin_with_distance_loss = TripletMarginWithDistanceLoss(reduction='mean')
            >>> loss = triplet_margin_with_distance_loss(input, positive, negative,)
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.19165532)

    """

    margin: float
    swap: bool
    reduction: _ReduceMode
    distance_function: Callable[[Tensor, Tensor], Tensor] | None
    name: str | None

    def __init__(
        self,
        distance_function: Callable[[Tensor, Tensor], Tensor] | None = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: _ReduceMode = 'mean',
        name: str | None = None,
    ) -> None:
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in TripletMarginWithDistanceLoss "
                "should be 'sum', 'mean' or 'none', but "
                f"received {reduction}, which is not allowed."
            )
        self.margin = margin
        self.swap = swap
        self.reduction = reduction
        self.distance_function = distance_function
        self.name = name

    def forward(
        self, input: Tensor, positive: Tensor, negative: Tensor
    ) -> Tensor:
        return F.triplet_margin_with_distance_loss(
            input,
            positive,
            negative,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction,
            name=self.name,
        )


class TripletMarginLoss(Layer):
    r"""
    Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `input`, `positive` and `negative` (i.e., `input`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, *)`.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(input, pos, neg) = \max \{d(input_i, pos_i) - d(input_i, neg_i) + {\rm margin}, 0\}


    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    Parameters:
        margin (float, Optional):Default: :math:`1`.

        p (float, Optional):The norm degree for pairwise distance. Default: :math:`2`.

        epsilon (float, Optional):Add small value to avoid division by zero,
            default value is 1e-6.

        swap (bool, Optional):The distance swap change the negative distance to the distance between
            positive sample and negative sample. For more details, see `Learning shallow convolutional feature descriptors with triplet losses`.
            Default: ``False``.

        reduction (str, Optional):Indicate how to average the loss by batch_size.
                the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
                If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
                If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
                If :attr:`reduction` is ``'sum'``, the summed loss is returned.
                Default: ``'mean'``

        name (str|None, Optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.

    Call Parameters:
        input (Tensor):Input tensor, the data type is float32 or float64.
        the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.

        positive (Tensor):Positive tensor, the data type is float32 or float64.
        The shape of label is the same as the shape of input.

        negative (Tensor):Negative tensor, the data type is float32 or float64.
        The shape of label is the same as the shape of input.

    Returns:
        Tensor. The tensor variable storing the triplet_margin_loss of input and positive and negative.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
            >>> positive= paddle.to_tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]], dtype=paddle.float32)
            >>> negative = paddle.to_tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]], dtype=paddle.float32)
            >>> triplet_margin_loss = paddle.nn.TripletMarginLoss(reduction='none')
            >>> loss = triplet_margin_loss(input, positive, negative)
            >>> print(loss)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.        , 0.57496595, 0.        ])

            >>> triplet_margin_loss = paddle.nn.TripletMarginLoss(margin=1.0, swap=True, reduction='mean')
            >>> loss = triplet_margin_loss(input, positive, negative)
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            2.40039468)

    """

    margin: float
    p: float
    epsilon: float
    swap: bool
    reduction: _ReduceMode
    name: str | None

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        epsilon: float = 1e-06,
        swap: bool = False,
        reduction: _ReduceMode = 'mean',
        name: str | None = None,
    ) -> None:
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in TripletMarginLoss should be 'sum', 'mean' or 'none', but "
                f"received {reduction}, which is not allowed."
            )
        self.margin = margin
        self.p = p
        self.epsilon = epsilon
        self.swap = swap
        self.reduction = reduction
        self.name = name

    def forward(
        self, input: Tensor, positive: Tensor, negative: Tensor
    ) -> Tensor:
        return F.triplet_margin_loss(
            input,
            positive,
            negative,
            margin=self.margin,
            p=self.p,
            epsilon=self.epsilon,
            swap=self.swap,
            reduction=self.reduction,
            name=self.name,
        )


class MultiMarginLoss(Layer):
    r"""Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between
    input :math:`input` and label :math:`label`:

    For i-th mini-batch sample, the loss in terms of the 1D input :math:`input_i` and scalar
    output :math:`label_i` is:

    .. math::
        \text{loss}(input_i, label_i) = \frac{\sum_{j} \max(0, \text{margin} - input_i[label_i] + input_i[j])^p}{\text{C}}

    where :math:`0 \leq j \leq \text{C}-1`, :math:`0 \leq i \leq \text{N}-1` and :math:`j \neq label_i`.

    Optionally, you can give non-equal weighting on the classes by passing
    a 1D :attr:`weight` tensor into the constructor.

    The loss function for i-th sample then becomes:

    .. math::
        \text{loss}(input_i, label_i) = \frac{\sum_{j} \max(0, weight[label_i] * (\text{margin} - input_i[label_i] + input_i[j]))^p}{\text{C}}


    Parameters:

        p (int, Optional):The norm degree for pairwise distance. Default: :math:`1`.

        margin (float, Optional):Default: :math:`1`.

        weight (Tensor,optional): a manual rescaling weight given to each class.
                If given, has to be a Tensor of shape (C,) and the data type is float32, float64.
                Default is ``'None'`` .

        reduction (str, optional): Indicate how to calculate the loss by batch_size,
                the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
                If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
                If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
                If :attr:`reduction` is ``'sum'``, the summed loss is returned.
                Default: ``'mean'``

        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Call parameters:
        input (Tensor): Input tensor, the data type is float32 or float64.

        label (Tensor): Label tensor, 0<= label < input.shape[1], the data type is int32 or int64.

    Shape:
        input: 2-D Tensor, the shape is [N, C], N is batch size and `C` means number of classes.

        label: 1-D Tensor, the shape is [N,].

        output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the label.

    Returns:
        A callable object of MultiMarginLoss.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn

            >>> input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
            >>> label = paddle.to_tensor([0, 1, 2], dtype=paddle.int32)

            >>> multi_margin_loss = nn.MultiMarginLoss(reduction='mean')
            >>> loss = multi_margin_loss(input, label)
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.11111104)
    """

    p: int
    margin: float
    weight: Tensor | None
    reduction: _ReduceMode
    name: str | None

    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight: Tensor | None = None,
        reduction: _ReduceMode = 'mean',
        name: str | None = None,
    ) -> None:
        super().__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "'reduction' in 'MultiMarginLoss' should be 'sum', 'mean' or 'none', "
                f"but received {reduction}."
            )
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction
        self.name = name

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        return F.multi_margin_loss(
            input,
            label,
            p=self.p,
            margin=self.margin,
            weight=self.weight,
            reduction=self.reduction,
            name=self.name,
        )


class SoftMarginLoss(Layer):
    r"""

    Creates a criterion that measures a two-class soft margin loss between input predictions ``input``
    and target labels ``label`` . It can be described as:

    .. math::
        Out = log(1 + exp((-label * input)))

    Parameters:

        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.

        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shapes:
        - Input (Tensor): The input tensor with shape: ``[N, *]``,
          N is batch_size, `*` means any number of additional dimensions. The ``input`` ranges from -inf to inf
          Available dtype is float32, float64.
        - Label (Tensor): The target labels tensor with the same shape as
          ``input``. The target labels which values should be numbers -1 or 1.
          Available dtype is int32, int64, float32, float64.
        - Output (Tensor): If ``reduction`` is ``'none'``, the shape of output is
          same as ``input`` , else the shape of output is [].

    Returns:
        A callable object of SoftMarginLoss.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> input = paddle.to_tensor([[0.5, 0.6, 0.7],[0.3, 0.5, 0.2]], 'float32')
            >>> label = paddle.to_tensor([[1.0, -1.0, 1.0],[-1.0, 1.0, 1.0]], 'float32')
            >>> soft_margin_loss = paddle.nn.SoftMarginLoss()
            >>> output = soft_margin_loss(input, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.64022040)

            >>> input_np = paddle.uniform(shape=(5, 5), min=0.1, max=0.8, dtype="float64")
            >>> label_np = paddle.randint(high=2, shape=(5, 5), dtype="int64")
            >>> label_np[label_np==0]=-1
            >>> input = paddle.to_tensor(input_np)
            >>> label = paddle.to_tensor(label_np)
            >>> soft_margin_loss = paddle.nn.SoftMarginLoss(reduction='none')
            >>> output = soft_margin_loss(input, label)
            >>> print(output)
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
            [[1.10725628, 0.48778139, 0.56217249, 1.12581404, 0.51430043],
             [0.90375795, 0.37761249, 0.43007557, 0.95089798, 0.43288319],
             [1.16043599, 0.63015939, 0.51362715, 0.43617541, 0.57783301],
             [0.81927846, 0.52558369, 0.59713908, 0.83100696, 0.50811616],
             [0.82684205, 1.02064907, 0.50296995, 1.13461733, 0.93222519]])
    """

    reduction: _ReduceMode
    name: str | None

    def __init__(
        self, reduction: _ReduceMode = 'mean', name: str | None = None
    ) -> None:
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in SoftMarginLoss should be 'sum', 'mean' or 'none', but "
                f"received {reduction}, which is not allowed."
            )

        super().__init__()
        self.reduction = reduction
        self.name = name

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        out = paddle.nn.functional.soft_margin_loss(
            input, label, self.reduction, self.name
        )
        return out


class GaussianNLLLoss(Layer):
    r"""Create a callable object of 'GaussianNLLLoss' to calculate Gaussian negative log likelihood loss.

    This class create a callable object of Gaussian negative log likelihood loss among ``input``, ``variance`` and
    ``label``. Note that the ``label`` is treated as samples from Gaussian distributions.
    This class is used to train a neural network predicts
    the ``input`` and ``variance`` of a gaussian distribution that ``label`` are supposed to
    be coming from. This means ``input`` and ``variance`` should be functions(the neural network) of some inputs.

    For a ``label`` having Gaussian distribution with ``input`` and ``variance`` predicted by neural network
    the loss is calculated as follows:

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{input} - \text{label}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

    where :attr:`epsilon` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``variance`` is not the same
    size as ``input`` (due to a homoscedastic assumption), it must either have a final dimension
    of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting.

    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``, means omit the constant term.
        epsilon (float, optional): value used to clamp ``variance`` (see note below), for
            stability. Default: 1e-6.
        reduction (str, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - Input(Tensor): :math:`(N, *)` or :math:`(*)` where :math:`*` means any number of additional
          dimensions. Available dtype is float32, float64.
        - Label(Tensor): :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input
          but with one dimension equal to 1 (to allow for broadcasting). Available dtype is float32, float64.
        - Variance(Tensor): :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting). Available dtype is float32, float64.
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Returns:
        A callable object of GaussianNLLLoss.

    Examples::
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn
            >>> paddle.seed(2023)

            >>> input = paddle.randn([5, 2], dtype=paddle.float32)
            >>> label = paddle.randn([5, 2], dtype=paddle.float32)
            >>> variance = paddle.ones([5, 2], dtype=paddle.float32)

            >>> gs_nll_loss = nn.GaussianNLLLoss(full=False, epsilon=1e-6, reduction='none')
            >>> loss = gs_nll_loss(input, label, variance)
            >>> print(loss)
            Tensor(shape=[5, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.21808575, 1.43013096],
             [1.05245590, 0.00394560],
             [1.20861185, 0.00000062],
             [0.56946373, 0.73300570],
             [0.37142906, 0.12038800]])

    Note:
        The clamping of ``variance`` is ignored with respect to autograd, and so the
        gradients are unaffected by it.
    """

    full: bool
    epsilon: float
    reduction: _ReduceMode
    name: str | None

    def __init__(
        self,
        full: bool = False,
        epsilon: float = 1e-06,
        reduction: _ReduceMode = 'mean',
        name: str | None = None,
    ) -> None:
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in GaussianNLLLoss should be 'sum', 'mean' or 'none', but "
                f"received {reduction}, which is not allowed."
            )

        super().__init__()
        self.full = full
        self.epsilon = epsilon
        self.reduction = reduction
        self.name = name

    def forward(self, input: Tensor, label: Tensor, variance: Tensor) -> Tensor:
        out = F.gaussian_nll_loss(
            input,
            label,
            variance,
            self.full,
            self.epsilon,
            self.reduction,
            self.name,
        )
        return out


class AdaptiveLogSoftmaxWithLoss(Layer):
    r"""Adaptive softmax is an approximate strategy for training models with large output spaces. It is most effective when
    the label distribution is highly imbalanced, for example in natural language modelling, where the word frequency
    distribution approximately follows the `Zipf's law <https://en.wikipedia.org/wiki/Zipf%27s_law>`_.

    Adaptive softmax partitions the labels into several clusters, according to their frequency. These clusters may contain
    different number of targets each. Additionally, clusters containing less frequent labels assign lower dimensional
    embeddings to those labels, which speeds up the computation. For each minibatch, only clusters for which at least
    one target is present are evaluated.

    The idea is that the clusters which are accessed frequently (like the first one, containing most frequent labels),
    should also be cheap to compute -- that is, contain a small number of assigned labels. We highly recommend taking
    a look at the original paper for more details.

    For :attr:`cutoffs` should be an ordered Sequence of integers sorted in the increasing order. It controls number of
    clusters and the partitioning of targets into clusters. For example setting ``cutoffs = [10, 100, 1000]`` means that
    first ``10`` targets will be assigned to the 'head' of the adaptive softmax, targets ``11, 12, ..., 100`` will be assigned
    to the first cluster, and targets ``101, 102, ..., 1000`` will be assigned to the second cluster, while targets
    ``1001, 1002, ..., n_classes - 1`` will be assigned to the last, third cluster.

    For :attr:`div_value` is used to compute the size of each additional cluster, which is given as follow:

    .. math::
        \lfloor \frac{\text{in\_features}}{\text{div\_value}^{idx}} \rfloor

    where :math:`idx` is the cluster index (with clusters for less frequent words having larger indices, and indices starting from :math:`1`).

    For :attr:`head_bias` if set to True, adds a bias term to the 'head' of the adaptive softmax. See paper for details. Set to False in the official implementation.


    Args:
        in_features (int): Number of features in the input tensor.
        n_classes (int): Number of classes in the dataset.
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets.
        weight_attr (ParamAttr, optional): The attribute for the learnable
            weight of this layer. The default value is None. If the Initializer of the
            param_attr is not set, the parameter is initialized with Xavier.
            For detailed information, please refer to :ref:`api_paddle_ParamAttr`.
        bias_attr (ParamAttr|bool|None, optional): The attribute for the learnable bias
            of this layer. If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to :ref:`api_paddle_ParamAttr`. The default value is None and the bias will be
            initialized to zero.
        div_value (float, optional): value used as an exponent to compute sizes of the clusters. Default: 4.0.
        head_bias (bool, optional): If ``True``, adds a bias term to the 'head' of the adaptive softmax. Default: ``False``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input (Tensor): The input tensor. The shapes is ``[N, in_features]``. N is batch size.
        - label (Tensor): target. The shapes is ``[N]``
        - output1 (Tensor): The shape is ``[N]``
        - output2 (Scalar).

    Returns:
        A callable object of AdaptiveLogSoftmaxWithLoss.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn
            >>> paddle.seed(2024)

            >>> input = paddle.randn([3, 5], dtype="float32")
            >>> target = paddle.full((3,), 1, dtype='int64')
            >>> asfm = nn.AdaptiveLogSoftmaxWithLoss(
            ...     in_features=5,
            ...     n_classes=3,
            ...     cutoffs=[2],
            ...     div_value=2.0,
            ...     head_bias=False
            ... )
            >>> out, loss = asfm(input, target)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=False,
            [-1.04691017, -0.42341536, -1.16909981])
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=False,
            0.87980843)
            >>> out = asfm.log_prob(input)
            >>> print(out)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[-1.13710010, -1.04691017, -1.11403584],
            [-1.51841831, -0.42341536, -2.07040048],
            [-4.25405550, -1.16909981, -0.39282480]])
            >>> out = asfm.predict(input)
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1., 1., 2.])

    Note:
        Labels passed as inputs to this module should be sorted according to their frequency. This means that the most
        frequent label should be represented by the index ``0``, and the least frequent label should be represented by
        the index ``n_classes - 1``. To compute log-probabilities for all classes, the ``log_prob`` method can be used.
    """

    in_features: int
    n_classes: int
    cutoffs: Sequence[int]
    weight_attr: ParamAttrLike | None
    bias_attr: ParamAttrLike | None
    div_value: float
    head_bias: Tensor | None
    head_weight: Tensor
    tail_weights: list[list[Tensor]]
    name: str | None

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: Sequence[int],
        weight_attr: ParamAttrLike | None = None,
        bias_attr: ParamAttrLike | None = None,
        div_value: float = 4.0,
        head_bias: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()
        cutoffs = list(cutoffs)

        if (
            (cutoffs != sorted(cutoffs))
            or (min(cutoffs) <= 0)
            or (max(cutoffs) > (n_classes - 1))
            or (len(set(cutoffs)) != len(cutoffs))
            or any(int(c) != c for c in cutoffs)
        ):
            raise ValueError(
                "cutoffs should be a sequence of unique, positive "
                "integers sorted in an increasing order, where "
                "each value is between 1 and n_classes-1"
            )

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = [*cutoffs, n_classes]
        self.div_value = div_value
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.is_head_bias = head_bias

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head_weight = self.create_parameter(
            shape=[self.in_features, self.head_size],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        if self.is_head_bias:
            self.head_bias = self.create_parameter(
                shape=[self.head_size],
                attr=self._bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
        else:
            self.head_bias = None

        self.tail_weights = []

        for i in range(self.n_clusters):
            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]
            projection = []
            projection.append(
                self.create_parameter(
                    shape=[self.in_features, hsz],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
            )
            projection.append(
                self.create_parameter(
                    shape=[hsz, osz],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
            )
            self.tail_weights.append(projection)

    def forward(self, input: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
        return F.adaptive_log_softmax_with_loss(
            input,
            label,
            self.head_weight,
            self.tail_weights,
            self.cutoffs,
            self.head_bias,
        )

    def _get_full_log_prob(self, input: Tensor, head_output: Tensor) -> Tensor:
        out = paddle.empty((head_output.shape[0], self.n_classes))
        head_logprob = F.log_softmax(head_output, axis=1)

        if paddle.in_dynamic_mode():
            out[:, : self.shortlist_size] = head_logprob[
                :, : self.shortlist_size
            ]
        else:
            paddle.static.setitem(
                out,
                (
                    slice(None, None, None),
                    slice(None, self.shortlist_size, None),
                ),
                head_logprob,
            )

        for i, (start_idx, stop_idx) in enumerate(
            zip(self.cutoffs, self.cutoffs[1:])
        ):
            cluster_output = F.linear(x=input, weight=self.tail_weights[i][0])
            cluster_output = F.linear(
                x=cluster_output, weight=self.tail_weights[i][1]
            )
            cluster_logprob = F.log_softmax(cluster_output, axis=1)
            output_logprob = cluster_logprob + head_logprob[
                :, self.shortlist_size + i
            ].unsqueeze(1)

            if paddle.in_dynamic_mode():
                out[:, start_idx:stop_idx] = output_logprob
            else:
                paddle.static.setitem(
                    out,
                    (slice(None, None, None), slice(start_idx, stop_idx, None)),
                    output_logprob,
                )

        return out

    def log_prob(self, input: Tensor) -> Tensor:
        head_output = F.linear(
            x=input, weight=self.head_weight, bias=self.head_bias
        )
        return self._get_full_log_prob(input, head_output)

    def predict(self, input: Tensor) -> Tensor:
        head_output = F.linear(
            x=input, weight=self.head_weight, bias=self.head_bias
        )
        output = paddle.argmax(head_output, axis=1).cast('float32')
        not_in_shortlist = output >= self.shortlist_size
        all_in_shortlist = not (not_in_shortlist.any())

        if all_in_shortlist:
            return output
        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(input, head_output)
            return paddle.argmax(log_prob, axis=1)
        else:
            log_prob = self._get_full_log_prob(
                input[not_in_shortlist], head_output[not_in_shortlist]
            )
            indices = paddle.masked_select(
                paddle.arange(len(not_in_shortlist)), not_in_shortlist
            )
            result = paddle.scatter(
                output, indices, paddle.argmax(log_prob, axis=1).cast('float32')
            )
            return result
