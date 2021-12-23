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

from __future__ import print_function

from paddle.fluid.clip import ClipGradBase, _squared_l2_norm
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid import core, layers, framework
from paddle.distributed import collective

import six
import warnings
import copy


class ClipGradForMOEByGlobalNorm(ClipGradBase):
    r"""
    The Algrithm is the same as paddle.fluid.clip.ClipGradByGlobalNorm
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
        Please use ``need_clip`` in ``ParamAttr`` to speficiy the clip scope.

    Args:
        clip_norm (float): The maximum norm value.
        moe_group (Group): group for moe experts communication.
        group_name (str, optional): The group name for this clip. Default value is ``default_moe_group``.

    Examples:
        .. code-block:: python
        
            import paddle

            x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
            linear = paddle.nn.Linear(in_features=10, out_features=10, 
                                      weight_attr=paddle.ParamAttr(need_clip=True), 
                                      bias_attr=paddle.ParamAttr(need_clip=False))
            out = linear(x)
            loss = paddle.mean(out)
            loss.backward()

            clip = paddle.nn.ClipGradForMOEByGlobalNorm(clip_norm=1.0,None)
            sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
            sdg.step()
    """

    def __init__(self,
                 clip_norm,
                 moe_group=None,
                 group_name="default_moe_group"):
        super(ClipGradForMOEByGlobalNorm, self).__init__()
        self.clip_norm = float(clip_norm)
        self.group_name = group_name
        self.moe_group = moe_group

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
                merge_grad = layers.merge_selected_rows(g)
                merge_grad = layers.get_tensor_from_selected_rows(merge_grad)

            sum_square = _squared_l2_norm(merge_grad)
            if sum_square.dtype == core.VarDesc.VarType.FP16:
                sum_square_list_fp16.append(sum_square)
            elif sum_square.dtype == core.VarDesc.VarType.FP32:
                sum_square_list_fp32.append(sum_square)
            else:
                sum_square_list.append(sum_square)

        # all parameters have been filterd out
        if len(sum_square_list) + len(sum_square_list_fp16) + len(
                sum_square_list_fp32) == 0:
            return None, None
        assert sum_dtype in ["float64", "float32", None], \
            "sum's type must be float64/ float32 / None"
        if sum_dtype != "float64":
            sum_dtype = 'float64' if len(sum_square_list) > 0 else "float32"

        global_norm_var = []
        if len(sum_square_list_fp16) > 0:
            global_norm_var_fp16 = layers.concat(sum_square_list_fp16)
            global_norm_var_fp16 = layers.reduce_sum(global_norm_var_fp16)
            global_norm_var.append(global_norm_var_fp16.astype(sum_dtype))
        if len(sum_square_list_fp32) > 0:
            global_norm_var_fp32 = layers.concat(sum_square_list_fp32)
            global_norm_var_fp32 = layers.reduce_sum(global_norm_var_fp32)
            if sum_dtype == 'float32':
                global_norm_var.append(global_norm_var_fp32)
            else:
                global_norm_var.append(global_norm_var_fp32.astype(sum_dtype))
        if len(sum_square_list) > 0:
            global_norm_var_fp64 = layers.concat(sum_square_list)
            global_norm_var_fp64 = layers.reduce_sum(global_norm_var_fp64)
            global_norm_var.append(global_norm_var_fp64)
        global_norm_var = layers.concat(global_norm_var)
        global_norm_var = layers.reduce_sum(global_norm_var)
        return global_norm_var, sum_dtype

    @imperative_base.no_grad
    def _dygraph_clip(self, params_grads):
        normal_params_grads = []
        moe_params_grads = []

        # seperate moe params from normal params
        if self.moe_group is not None and self.moe_group.nranks > 1:
            for p, g in params_grads:
                if "experts_" in p.name:
                    moe_params_grads.append((p, g))
                else:
                    normal_params_grads.append((p, g))
        else:
            normal_params_grads = params_grads

        global_norm_var_normal, sum_dtype \
            = self.get_l2_norm_pow(normal_params_grads)
        global_norm_var_moe = None
        if len(moe_params_grads) > 0:
            global_norm_var_moe, _ \
                = self.get_l2_norm_pow(moe_params_grads, sum_dtype)
            if global_norm_var_moe is not None:
                collective.all_reduce(
                    global_norm_var_moe,
                    op=collective.ReduceOp.SUM,
                    group=self.moe_group)

        if global_norm_var_normal is None and global_norm_var_moe is None:
            return params_grads
        elif global_norm_var_normal is None:
            global_norm_var = global_norm_var_moe
        elif global_norm_var_moe is None:
            global_norm_var = global_norm_var_normal
        else:
            global_norm_var = global_norm_var_normal + global_norm_var_moe

        params_and_grads = []
        global_norm_var = layers.sqrt(global_norm_var)
        max_global_norm = layers.fill_constant(
            shape=[1], dtype=global_norm_var.dtype, value=self.clip_norm)
        clip_var = layers.elementwise_div(
            x=max_global_norm,
            y=layers.elementwise_max(
                x=global_norm_var, y=max_global_norm))
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            # TODO(wangxi): use inplace elementwise_mul
            clip_input = (clip_var.astype('float16')
                          if g.dtype == core.VarDesc.VarType.FP16 else clip_var)
            new_grad = layers.elementwise_mul(x=g, y=clip_input)
            params_and_grads.append((p, new_grad))

        return params_and_grads

    def _process_context(self, context, param, grad):
        if self.group_name not in context:
            context[self.group_name] = []
            context[self.group_name + "_clip_value"] = self.clip_norm
            context[self.group_name + "_clip"] = layers.fill_constant(
                shape=[1], dtype=grad.dtype, value=self.clip_norm)
        else:
            if not self.clip_norm == context[self.group_name + "_clip_value"]:
                raise ValueError(
                    "All parameters' 'clip_norm' of a same group should be the same"
                )

        merge_grad = grad
        if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
            merge_grad = layers.merge_selected_rows(grad)
            merge_grad = layers.get_tensor_from_selected_rows(merge_grad)

        local_norm_var = _squared_l2_norm(merge_grad)
        context[self.group_name].append(local_norm_var)

        self.context = context

    def _create_operators(self, param, grad):
        group_scale_name = self.group_name + "_scale"
        if group_scale_name not in self.context:
            group_norm_var = layers.sums(input=self.context[self.group_name])
            group_norm_var = layers.sqrt(x=group_norm_var)
            clip_var = self.context[self.group_name + "_clip"]
            group_scale_var = layers.elementwise_div(
                x=clip_var,
                y=layers.elementwise_max(
                    x=clip_var, y=group_norm_var))
            assert group_scale_var.shape == (1, )
            self.context[group_scale_name] = group_scale_var

        # inplace
        param.block.append_op(
            type='elementwise_mul',
            inputs={'X': grad,
                    'Y': self.context[group_scale_name]},
            outputs={'Out': grad})

        return param, grad


@framework.dygraph_not_support
def set_gradient_clip(clip, param_list=None, program=None):
    """
    :api_attr: Static Graph
    
    Warning:
    
        This API must be used after building network, and before ``minimize`` , 
        and it may be removed in future releases, so it is not recommended. 
        It is recommended to set ``grad_clip`` when initializing the ``optimizer`` ,
        this is a better method to clip gradient. There are three clipping strategies:
         :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
         :ref:`api_fluid_clip_GradientClipByValue` .
        
    To specify parameters that require gradient clip.

    Args:
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default value: None, and there is no 
            gradient clipping.
        param_list (list(Variable), optional): Parameters that require gradient clip.
                It can be a list of parameter or a list of parameter's name.
                Default None, meaning that all parameters in the program will be included.
        program (Program, optional): The program where parameters are located.
                Default None, meaning that using :ref:`api_fluid_default_main_program` .

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            def network():
                image = fluid.data(name='image', shape=[
                                   None, 28], dtype='float32')
                param_attr1 = fluid.ParamAttr("fc1_param")
                fc1 = fluid.layers.fc(image, size=10, param_attr=param_attr1)
                param_attr2 = fluid.ParamAttr("fc2_param")
                fc2 = fluid.layers.fc(fc1, size=10, param_attr=param_attr2)
                loss = fluid.layers.reduce_mean(fc2)
                return loss


            # network 1: clip all parameter gradient
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                loss = network()
                fluid.clip.set_gradient_clip(
                    fluid.clip.GradientClipByGlobalNorm(clip_norm=2.0))
                sgd = fluid.optimizer.SGD(learning_rate=1e-3)
                sgd.minimize(loss)

            # network 2: clip parameter gradient by name
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                loss = network()
                fluid.clip.set_gradient_clip(
                    fluid.clip.GradientClipByValue(min=-1.0, max=1.0),
                    param_list=["fc1_param", "fc2_param"])
                sgd = fluid.optimizer.SGD(learning_rate=1e-3)
                sgd.minimize(loss)

            # network 3: clip parameter gradient by value
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                loss = network()
                param_var1 = fluid.default_main_program().global_block().var("fc1_param")
                param_var2 = fluid.default_main_program().global_block().var("fc2_param")
                fluid.clip.set_gradient_clip(
                    fluid.clip.GradientClipByValue(min=-1.0, max=1.0),
                    param_list=[param_var1, param_var2])
                sgd = fluid.optimizer.SGD(learning_rate=1e-3)
                sgd.minimize(loss)
            
            # network 4: use 'set_gradient_clip' and 'optimize(grad_clip=clip)' together
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                loss = network()
                clip1 = fluid.clip.GradientClipByValue(min=-1.0, max=1.0)
                clip2 = fluid.clip.GradientClipByNorm(clip_norm=1.0)
                # Set the gradient clipping strategy: clip1
                fluid.clip.set_gradient_clip(clip1)
                # Set the gradient clipping strategy: clip2
                sgd = fluid.optimizer.SGD(learning_rate=1e-3, grad_clip=clip2)
                sgd.minimize(loss)
                # 'set_gradient_clip' will not take effect when setting has a conflict, 
                # and the gradient clipping strategy will be 'clip2'
            
            
    """
    warnings.warn("Caution! 'set_gradient_clip' is not recommended "
                  "and may be deprecated in future! "
                  "We recommend a new strategy: set 'grad_clip' "
                  "when initializing the 'optimizer'. "
                  "This method can reduce the mistakes, please "
                  "refer to documention of 'optimizer'.")

    if not isinstance(clip, ClipGradBase):
        raise TypeError(
            "'clip' should be an instance of ClipGradBase's derived class")
    if program is None:
        program = framework.default_main_program()

    for op in program.block(0).ops:
        if 'op_namescope' in op.all_attrs() and "optimizer" in op.attr(
                "op_namescope"):
            warnings.warn(
                "'minimize' has been invoked before, this will make 'set_gradient_clip' "
                "be ineffective! Please invoke 'set_gradient_clip' before 'minimize'."
            )
            break

    if param_list is None:
        param_list = program.block(0).all_parameters()
    if all(isinstance(elem, six.string_types) for elem in param_list):
        param_list = [program.block(0).var(elem) for elem in param_list]
    if not all(isinstance(elem, framework.Parameter) for elem in param_list):
        raise TypeError(
            "'param_list' should be a list of Parameter or basestring(parameter's name)."
        )

    for param in param_list:
        param.gradient_clip_attr = copy.deepcopy(clip)


ClipGradByGlobalNorm = ClipGradForMOEByGlobalNorm

__all__ = ['ClipGradByGlobalNorm']
