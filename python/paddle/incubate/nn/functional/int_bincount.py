import paddle
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import convert_dtype


def int_bincount(x, low, high, dtype=None, name=None):
    if in_dynamic_or_pir_mode():
        return _C_ops.moe_gate_dispatch_permute(x, gate_logits, corr_bias, k, capacity, world_size)
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