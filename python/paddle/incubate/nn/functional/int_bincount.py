import paddle
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import convert_dtype


def int_bincount(x, low, high, dtype=None, name=None):
    """
    对 1-D 整数 Tensor x 做 bincount。
    Args:
      x (Tensor): 1-D Tensor，类型为 int32/int64。
      low (int): 直方图下界（inclusive）。
      high (int): 直方图上界（inclusive）。
      dtype (np.dtype|str, optional): 输出 Tensor 的数据类型，默认与 x.dtype 相同。
      name (str, optional): Op 名称。
    Returns:
      Tensor y: 形状 [high - low + 1] 的计数结果，dtype 为 dtype。
    """
    helper = LayerHelper("int_bincount", **locals())
    # 决定输出 dtype
    out_dtype = dtype if dtype is not None else x.dtype
    # 创建输出变量
    y = helper.create_variable_for_type_inference(dtype=out_dtype)

    # convert_dtype 将 paddle dtype 转为框架里标量值（int64_t）
    dtype_attr = convert_dtype(out_dtype)

    helper.append_op(
        type="int_bincount",
        inputs={"x": x},
        outputs={"y": y},
        attrs={
            "low":    low,
            "high":   high,
            "dtype":  dtype_attr,
        })
    return y
