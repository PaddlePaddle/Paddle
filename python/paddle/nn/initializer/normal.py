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

from paddle import _C_ops

from ...fluid import core, framework, unique_name
from ...fluid.data_feeder import check_variable_and_dtype
from ...fluid.framework import _current_expected_place, in_dygraph_mode
from .initializer import Initializer

__all__ = []


class NormalInitializer(Initializer):
    """Implements the Random Normal(Gaussian) distribution initializer

    Args:
        loc (float, optional): mean of the normal distribution. Default is 0.0.
        scale (float, optional): standard deviation of the normal distribution. Default is 1.0.
        seed (int, optional): random seed. Default is 0.

    """

    def __init__(self, loc=0.0, scale=1.0, seed=0):
        assert loc is not None
        assert scale is not None
        assert seed is not None
        super().__init__()
        self._mean = loc
        self._std_dev = scale
        self._seed = seed

    def forward(self, var, block=None):
        """Initialize the input tensor with Normal distribution.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(block, framework.Block)

        check_variable_and_dtype(
            var,
            "Out",
            ["uint16", "float16", "float32", "float64"],
            "guassian_random",
        )

        if self._seed == 0:
            self._seed = block.program.random_seed

        if in_dygraph_mode():
            place = _current_expected_place()
            out_var = _C_ops.gaussian(
                var.shape,
                self._mean,
                self._std_dev,
                self._seed,
                var.dtype,
                place,
            )
            out_var._share_underline_tensor_to(var)
            return None

        else:
            op = block.append_op(
                type="gaussian_random",
                outputs={"Out": var},
                attrs={
                    "shape": var.shape,
                    "dtype": var.dtype,
                    "mean": self._mean,
                    "std": self._std_dev,
                    "seed": self._seed,
                    "use_mkldnn": False,
                },
                stop_gradient=True,
            )
            var.op = op
            return op


class Normal(NormalInitializer):
    """The Random Normal (Gaussian) distribution initializer.

    Args:
        mean (float, optional): mean of the normal distribution. Default is 0.0.
        std (float, optional): standard deviation of the normal distribution. Default is 1.0.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        A parameter initialized by Random Normal (Gaussian) distribution.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr = paddle.framework.ParamAttr(
                name="linear_weight",
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0))
            bias_attr = paddle.framework.ParamAttr(
                name="linear_bias",
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0))
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            # linear.weight:  [[ 2.1973135 -2.2697184]
            #                  [-1.9104223 -1.0541488]]
            # linear.bias:  [ 0.7885926  -0.74719954]

            res = linear(data)
            # res:  [[[ 1.0754838 -4.071067 ]]
            #        [[ 1.0754838 -4.071067 ]]
            #        [[ 1.0754838 -4.071067 ]]]
    """

    def __init__(self, mean=0.0, std=1.0, name=None):
        assert mean is not None, 'mean should not be None'
        assert std is not None, 'std should not be None'
        super().__init__(loc=mean, scale=std, seed=0)


class TruncatedNormalInitializer(Initializer):
    """Implements the Random TruncatedNormal(Gaussian) distribution initializer

    Args:
        loc (float, optional): Mean of the normal distribution. Default is :math:`0.0`.
        scale (float, optional): Standard deviation of the normal distribution. Default is :math:`1.0`.
        seed (int, optional): random seed. Default is 0.

    """

    def __init__(self, loc=0.0, scale=1.0, seed=0):
        assert loc is not None
        assert scale is not None
        assert seed is not None
        super().__init__()
        self._mean = loc
        self._std_dev = scale
        self._seed = seed

    def forward(self, var, block=None):
        """Initialize the input tensor with TruncatedNormal distribution.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype in [core.VarDesc.VarType.FP16, core.VarDesc.VarType.BF16]:
            out_dtype = core.VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(
                    ".".join(['truncated_gaussian_random', var.name, 'tmp'])
                ),
                shape=var.shape,
                dtype=out_dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
            )
        else:
            out_dtype = var.dtype
            out_var = var

        if in_dygraph_mode():
            out_var = _C_ops.truncated_gaussian_random(
                var.shape,
                self._mean,
                self._std_dev,
                self._seed,
                out_dtype,
                _current_expected_place(),
            )
            if var.dtype in [
                core.VarDesc.VarType.FP16,
                core.VarDesc.VarType.BF16,
            ]:
                var_tmp = _C_ops.cast(out_var, var.dtype)
                var_tmp._share_underline_tensor_to(var)
            else:
                out_var._share_underline_tensor_to(var)
            return None

        else:
            op = block.append_op(
                type="truncated_gaussian_random",
                outputs={"Out": out_var},
                attrs={
                    "shape": var.shape,
                    "dtype": out_dtype,
                    "mean": self._mean,
                    "std": self._std_dev,
                    "seed": self._seed,
                },
                stop_gradient=True,
            )

            if var.dtype in [
                core.VarDesc.VarType.FP16,
                core.VarDesc.VarType.BF16,
            ]:
                block.append_op(
                    type="cast",
                    inputs={"X": out_var},
                    outputs={"Out": var},
                    attrs={"in_dtype": out_var.dtype, "out_dtype": var.dtype},
                )
            var.op = op
            return op


class TruncatedNormal(TruncatedNormalInitializer):
    """The truncated normal distribution (Gaussian distribution) initializer.

    Args:
        mean (float, optional): Mean of the normal distribution. Default is :math:`0.0`.
        std (float, optional): Standard deviation of the normal distribution. Default is :math:`1.0`.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        A parameter initialized by truncated normal distribution (Gaussian distribution).

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr = paddle.framework.ParamAttr(
                name="linear_weight",
                initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=2.0))
            bias_attr = paddle.framework.ParamAttr(
                name="linear_bias",
                initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=2.0))
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            # linear.weight:  [[-1.0981836  1.4140984]
            #                  [ 3.1390522 -2.8266568]]
            # linear.bias:  [-2.1546738 -1.6570673]

            res = linear(data)
            # res:  [[[-0.11380529 -3.0696259 ]]
            #        [[-0.11380529 -3.0696259 ]]
            #        [[-0.11380529 -3.0696259 ]]
    """

    def __init__(self, mean=0.0, std=1.0, name=None):
        assert mean is not None, 'mean should not be None'
        assert std is not None, 'std should not be None'
        super().__init__(loc=mean, scale=std, seed=0)
