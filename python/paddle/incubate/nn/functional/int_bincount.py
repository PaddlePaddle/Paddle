import paddle
from paddle import _C_ops
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper


def int_bincount(x, low, high, dtype=None, name=None):
    if in_dynamic_or_pir_mode():
        return _C_ops.int_bincount(x, low, high, dtype)
    
    helper = LayerHelper("int_bincount", **locals())
    out_dtype = dtype if dtype is not None else x.dtype
    y = helper.create_variable_for_type_inference(dtype=out_dtype)
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