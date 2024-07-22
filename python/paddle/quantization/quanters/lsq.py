import paddle
from paddle import _C_ops

def fake_lsq_quant_dequant(x, scale, lsq_factor = 1.0, bit_length = 8, round_type = 1):
    def __check_input(x, scale, lsq_factor = 1.0, bit_length = 8, round_type = 1):
        assert round_type == 0 or round_type == 1

    __check_input(x, scale, lsq_factor, bit_length, round_type)

    # if paddle.in_dygraph_mode():
    return _C_ops.fake_quantize_dequantize_lsq( x, scale, lsq_factor, bit_length, round_type )

    helper = LayerHelper('fake_quantize_dequantize_lsq', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='fake_quantize_dequantize_lsq',
        inputs={'Input': [x, scale]},
        attrs={'lsq_factor': lsq_factor,
               'bit_length': bit_length,
               'round_type': round_type},
        outputs={'Out': [out]})
    return out