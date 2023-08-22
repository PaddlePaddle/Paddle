# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.common_ops_import import default_main_program
from paddle.fluid import core
from paddle.framework import LayerHelper, in_dynamic_mode


def fused_dropout_add(
    x, y, p=0.5, training=True, mode='upscale_in_train', name=None
):
    r"""
    Fused Dropout and Add.

    Args:
        x (Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64.
        y (Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64.

        p (float|int, optional): Probability of setting units to zero. Default: 0.5.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default: True.
        mode(str, optional): ['upscale_in_train'(default) | 'downscale_in_infer'].

            1. upscale_in_train (default), upscale the output at training time

                - train: :math:`out = x \times \frac{mask}{(1.0 - dropout\_prob)} + y`
                - inference: :math:`out = x + y`

            2. downscale_in_infer, downscale the output at inference

                - train: :math:`out = input \times mask + y`
                - inference: :math:`out = input \times (1.0 - dropout\_prob) + y`

        name (str, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor representing the fused dropout and add, has same shape and data type as `x` .


    Examples:

        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.seed(2023)
            >>> from paddle.incubate.nn.functional import fused_dropout_add

            >>> x = paddle.randn([4, 10], dtype='float16')
            >>> y = paddle.randn([4, 10], dtype='float16')
            >>> out = fused_dropout_add(x, y, p=0.5)
            >>> print(out)
            Tensor(shape=[4, 10], dtype=float16, place=Place(gpu:0), stop_gradient=True,
            [[-0.49121094,  0.53808594, -2.58398438,  0.06347656, -1.09960938,
               0.22082520,  2.19726562,  0.05053711,  0.53417969,  0.84863281],
             [ 0.78271484, -1.59667969, -0.14404297, -0.77929688, -0.17004395,
              -0.30981445, -0.36572266, -0.51025391,  1.46386719,  0.61621094],
             [ 4.50390625, -0.48461914,  0.60742188,  0.33496094, -0.25585938,
              -1.45214844,  1.06738281,  0.00439453, -0.77343750,  0.67382812],
             [ 1.29492188,  0.07568359,  0.71923828, -0.71777344, -2.57226562,
               1.89160156,  3.26367188,  1.10546875, -1.04589844, -1.04882812]])
    """
    if isinstance(p, (int, float)):
        # fast return for p == 0
        if p == 0:
            return x + y
        elif p < 0 or p > 1:
            raise ValueError("p argument should between 0 and 1")
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
        )
    seed = None
    if in_dynamic_mode():
        if default_main_program().random_seed != 0:
            seed = default_main_program().random_seed
        out, seed_offset = _C_ops.fused_dropout_add(
            x,
            y,
            None,
            p,
            not training,
            mode,
            seed if seed is not None else 0,
            seed is not None,
        )
        return out
    else:
        helper = LayerHelper('fused_dropout_add', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        seed_offset = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.INT64, stop_gradient=True
        )

        def get_attrs(prog, dropout_prob, is_test, seed):
            if (seed is None or seed == 0) and prog.random_seed != 0:
                seed = prog.random_seed
            attrs = {
                'p': dropout_prob,
                'is_test': is_test,
                'mode': mode,
                'seed': seed if seed is not None else 0,
                'fix_seed': seed is not None,
            }
            return attrs

        attrs = get_attrs(helper.main_program, p, not training, seed)

        helper.append_op(
            type='fused_dropout_add',
            inputs={'x': x, 'y': y, 'seed_tensor': None},
            outputs={'out': [out], 'seed_offset': [seed_offset]},
            attrs=attrs,
        )
        return out
