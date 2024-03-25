#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed as dist
from paddle.autograd import no_grad
from paddle.framework import core
from paddle.nn import clip
from paddle.nn.clip import ClipGradBase, _squared_l2_norm


class ClipGradForMOEByGlobalNorm(ClipGradBase):
    r"""
    The Algorithm is the same as paddle.nn.ClipGradByGlobalNorm
    Given a list of Tensor :math:`t\_list` , calculate the global norm for the elements of all tensors in
    :math:`t\_list` , and limit it to ``clip_norm`` .

    - If the global norm is greater than ``clip_norm`` , all elements of :math:`t\_list` will be compressed by a ratio.

    - If the global norm is less than or equal to ``clip_norm`` , nothing will be done.

    The list of Tensor :math:`t\_list` is not passed from this class, but the gradients of all parameters set in ``optimizer``.
    If ``need_clip`` of specific param is ``False`` in its ``ParamAttr``, then the gradients of this param will not be clipped.

    Gradient clip will takes effect after being set in ``optimizer`` , see the document ``optimizer``
    (for example: :ref:`api_paddle_optimizer_SGD`).

    The clipping formula is:

    .. math::

        t\_list[i] = t\_list[i] * \frac{clip\_norm}{\max(global\_norm, clip\_norm)}

    where:

    .. math::

        global\_norm = \sqrt{\sum_{i=0}^{N-1}(l2norm(t\_list[i]))^2}

    Note:
        ``need_clip`` of ``ClipGradyGlobalNorm`` HAS BEEN DEPRECATED since 2.0.
        Please use ``need_clip`` in ``ParamAttr`` to specify the clip scope.

    Reference:
        https://github.com/laekov/fastmoe/blob/master/examples/megatron/clip-grad-v2.2.patch
        Git commit hash: 295a615aacce7e54a37e7935274ba15e901c78e4


    Args:
        clip_norm (float): The maximum norm value.
        is_expert_param_func (function): a function to decide whether a param should be put into moe_params_grads
        moe_group (Group): group for moe experts communication.
        group_name (str, optional): The group name for this clip. Default value is ``default_moe_group``.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
            >>> linear = paddle.nn.Linear(in_features=10, out_features=10,
            ...                           weight_attr=paddle.ParamAttr(need_clip=True),
            ...                           bias_attr=paddle.ParamAttr(need_clip=False))
            >>> out = linear(x)
            >>> loss = paddle.mean(out)
            >>> loss.backward()

            >>> clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0) # Cause paddle.nn hasn't this interface, so we use ClipGradByGlobalNorm here.
            >>> sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
            >>> sdg.step()
    """

    def __init__(
        self,
        clip_norm,
        is_expert_param_func=None,
        moe_group=None,
        group_name="default_moe_group",
    ):
        super().__init__()
        self.clip_norm = float(clip_norm)
        self.group_name = group_name
        self.moe_group = moe_group
        if moe_group is not None and moe_group.nranks > 1:
            assert (
                is_expert_param_func is not None
            ), "When moe group size > 1, a function for selecting expert params must be specified."
        self.is_expert_param_func = is_expert_param_func

    def __str__(self):
        return "Gradient Clip By GlobalNorm, global_norm=%f" % (self.clip_norm)

    @staticmethod
    def get_l2_norm_pow(params_grads, sum_dtype=None):
        sum_square_list = []
        sum_square_list_fp16 = []
        sum_square_list_fp32 = []
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = clip.merge_selected_rows(g)
                merge_grad = clip.get_tensor_from_selected_rows(merge_grad)
            sum_square = _squared_l2_norm(merge_grad)
            if sum_square.dtype == paddle.float16:
                sum_square_list_fp16.append(sum_square)
            elif sum_square.dtype == paddle.float32:
                sum_square_list_fp32.append(sum_square)
            else:
                sum_square_list.append(sum_square)

        # all parameters have been filtered out
        if (
            len(sum_square_list)
            + len(sum_square_list_fp16)
            + len(sum_square_list_fp32)
            == 0
        ):
            return None, None
        assert sum_dtype in [
            "float64",
            "float32",
            None,
        ], "sum's type must be float64/ float32 / None"
        if sum_dtype != "float64":
            sum_dtype = 'float64' if len(sum_square_list) > 0 else "float32"

        global_norm_var = []
        if len(sum_square_list_fp16) > 0:
            global_norm_var_fp16 = paddle.add_n(sum_square_list_fp16)
            global_norm_var.append(global_norm_var_fp16.astype(sum_dtype))
        if len(sum_square_list_fp32) > 0:
            global_norm_var_fp32 = paddle.add_n(sum_square_list_fp32)
            if sum_dtype == 'float32':
                global_norm_var.append(global_norm_var_fp32)
            else:
                global_norm_var.append(global_norm_var_fp32.astype(sum_dtype))
        if len(sum_square_list) > 0:
            global_norm_var_fp64 = paddle.add_n(sum_square_list)
            global_norm_var.append(global_norm_var_fp64)
        global_norm_var = paddle.add_n(global_norm_var)
        return global_norm_var, sum_dtype

    @no_grad()
    def _dygraph_clip(self, params_grads):
        normal_params_grads = []
        moe_params_grads = []

        # separate moe params from normal params
        if self.moe_group is not None and self.moe_group.nranks > 1:
            for p, g in params_grads:
                if self.is_expert_param_func(p):
                    moe_params_grads.append((p, g))
                else:
                    normal_params_grads.append((p, g))
        else:
            normal_params_grads = params_grads

        # why to return sum_dtype?
        # we will call `get_l2_norm_pow` twice and the precisions may be different.
        # For convenience and simplification, we use sum_dtype directly instead of global_norm_var_normal.dtype
        global_norm_var_normal, sum_dtype = self.get_l2_norm_pow(
            normal_params_grads
        )
        global_norm_var_moe = None
        if len(moe_params_grads) > 0:
            global_norm_var_moe, _ = self.get_l2_norm_pow(
                moe_params_grads, sum_dtype
            )
            if global_norm_var_moe is not None:
                dist.all_reduce(
                    global_norm_var_moe,
                    op=dist.ReduceOp.SUM,
                    group=self.moe_group,
                )

        if global_norm_var_normal is None and global_norm_var_moe is None:
            return params_grads
        elif global_norm_var_normal is None:
            global_norm_var = global_norm_var_moe
        elif global_norm_var_moe is None:
            global_norm_var = global_norm_var_normal
        else:
            if global_norm_var_normal.dtype != global_norm_var_moe.dtype:
                # compared with normal norm, moe norm is the later one,
                # so its precision is no lower than normal norm
                global_norm_var_normal = global_norm_var_normal.astype(
                    global_norm_var_moe.dtype
                )
            global_norm_var = global_norm_var_normal + global_norm_var_moe

        params_and_grads = []
        global_norm_var = paddle.sqrt(global_norm_var)
        max_global_norm = paddle.full(
            shape=[1], dtype=global_norm_var.dtype, fill_value=self.clip_norm
        )
        clip_var = paddle.divide(
            x=max_global_norm,
            y=paddle.maximum(x=global_norm_var, y=max_global_norm),
        )
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            # TODO(wangxi): use inplace elementwise_mul
            clip_input = (
                clip_var.astype('float16')
                if g.dtype == paddle.float16
                else clip_var
            )
            new_grad = paddle.multiply(x=g, y=clip_input)
            params_and_grads.append((p, new_grad))
        return params_and_grads


ClipGradByGlobalNorm = ClipGradForMOEByGlobalNorm
