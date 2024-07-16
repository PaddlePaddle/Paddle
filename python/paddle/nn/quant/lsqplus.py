import paddle
import paddle

from paddle import _C_ops
from ..base.layer_helper import LayerHelper
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.layer_helper import LayerHelper

from paddle
def lsqplus(x, alpha, beta, g, Qn, Qp):
    """
    ***lsqplus***

    This is a Lsqplus quantization algorithm operator, which is implemented on kernel level.
    This operator supports both forward and backward computation.

    Args:
        x (Tensor): The input tensor of Lsqplus quantization operator.
        alpha (float): The stepsize parameter.
        beta (float): The bias parameter.
        g (float): The gradient scaling parameter.
        Qn (int): The quantization minimum value.
        Qp (int): The quantization maximum value.

    Returns:
        output (Tensor): The fakequantized tensor of Lsqplus quantization operator.
    """

    def __check_input(x, alpha, beta, g, Qn, Qp):

        # 类型检查
        check_variable_and_dtype(x, 'x', [paddle.Tensor], 'lsqplus', \
                                extra_message="The dtype of the input must a paddle.Tensor")
        check_variable_and_dtype(alpha, 'alpha', [paddle.Tensor], 'lsqplus', \
                                extra_message="The dtype of the alpha must a paddle.Tensor")
        check_variable_and_dtype(beta, 'beta', [paddle.Tensor], 'lsqplus', \
                                extra_message="The dtype of the beta must a paddle.Tensor")
        check_variable_and_dtype(g, 'g', [paddle.Tensor], 'lsqplus', \  
                                extra_message="The dtype of the g must a paddle.Tensor")
        
        check_variable_and_dtype(Qn, 'Qn', [int], 'lsqplus', \
                                extra_message="The dtype of the Qn must a int")
        check_variable_and_dtype(Qp, 'Qp', [int], 'lsqplus', \
                                extra_message="The dtype of the Qp must a int")

        # 维度检查
        assert len(x.shape) >= 1 and x.numel() >= 1, "the input's shape length should not be zero"
        assert len(alpha.shape) == 1 and alpha.numel() == 1, "the alpha should be a scalar"
        assert len(beta.shape) == 1 and beta.numel() == 1, "the beta should be a scalar"
        assert len(g.shape) == 1 and g.numel() == 1, "the g should be a scalar"

        # 数值检查
        assert len(x.shape) == 1, "the input's dimension should be one"
        assert Qn < Qp, "the min value should be less than max value"
    
    __check_input(x, alpha, beta, g, Qn, Qp)

    # 动态图分支
    if base.in_dygraph_mode():
        return _C_ops.lsqplus(x, alpha, beta, g, Qn, Qp)
    
    # 静态图分支
    helper = LayerHelper('lsqplus', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='lsqplus',
        inputs={'x': [x]
               'alpha': [alpha],
               'beta': [beta],
               'g': [g]} ,
        attrs={'Qn': Qn,
               'Qp': Qp},
        outputs={'out': [out]}
        )
    return out

