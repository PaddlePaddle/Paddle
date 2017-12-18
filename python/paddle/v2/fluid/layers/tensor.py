from ..layer_helper import LayerHelper

__all__ = [
    'create_tensor', 'cast', 'concat', 'sums', 'assign',
    'fill_constant_batch_size_like', 'fill_constant', 'ones', 'zeros'
]


def create_tensor(dtype, name=None, main_program=None, startup_program=None):
    helper = LayerHelper("create_tensor", **locals())
    return helper.create_variable(name=helper.name, dtype=dtype)


def cast(x, dtype, main_program=None):
    """
    This function takes in the input with input_dtype
    and casts it to the output_dtype as the output.
    """
    helper = LayerHelper('cast', **locals())
    out = helper.create_tmp_variable(dtype=dtype)
    helper.append_op(
        type='cast',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'in_dtype': x.dtype,
               'out_dtype': out.dtype})
    return out


def concat(input, axis, main_program=None, startup_program=None):
    """
    This function concats the input along the axis mentioned
    and returns that as the output.
    """
    helper = LayerHelper('concat', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='concat',
        inputs={'X': input},
        outputs={'Out': [out]},
        attrs={'axis': axis})
    return out


def sums(input, out=None, main_program=None, startup_program=None):
    """
    This function takes in the input and performs the sum operation on it
    and returns that as the output.
    """
    helper = LayerHelper('sum', **locals())
    if out is None:
        out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(type='sum', inputs={'X': input}, outputs={'Out': out})
    return out


def assign(input, output, main_program=None, startup_program=None):
    helper = LayerHelper('assign', **locals())
    helper.append_op(
        type='scale',
        inputs={'X': [input]},
        outputs={'Out': [output]},
        attrs={'scale': 1.0})
    return output


def fill_constant(shape,
                  dtype,
                  value,
                  out=None,
                  main_program=None,
                  startup_program=None):
    """
    This function creates a tensor , with shape as mentioned in the input and
    specified dtype and fills this up with a constant value that
    comes in the input. It also sets the stop_gradient to be True.
    """
    helper = LayerHelper("fill_constant", **locals())
    if out is None:
        out = helper.create_tmp_variable(dtype=dtype)
    helper.append_op(
        type='fill_constant',
        inputs={},
        outputs={'Out': [out]},
        attrs={'shape': shape,
               'dtype': out.dtype,
               'value': float(value)})
    out.stop_gradient = True
    return out


def fill_constant_batch_size_like(input,
                                  shape,
                                  dtype,
                                  value,
                                  input_dim_idx=0,
                                  output_dim_idx=0,
                                  main_program=None,
                                  startup_program=None):
    helper = LayerHelper("fill_constant_batch_size_like", **locals())
    out = helper.create_tmp_variable(dtype=dtype)
    helper.append_op(
        type='fill_constant_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': [out]},
        attrs={
            'shape': shape,
            'dtype': out.dtype,
            'value': float(value),
            'input_dim_idx': input_dim_idx,
            'output_dim_idx': output_dim_idx
        })
    out.stop_gradient = True
    return out


def ones(shape, dtype, main_program=None):
    """
    This function performs the same function as fill_constant() declared above
    with the constant value being 1.0.
    """
    return fill_constant(value=1.0, **locals())


def zeros(shape, dtype, main_program=None):
    """
    This function performs the same function as fill_constant() declared above
    with the constant value being 0.0.
    """
    return fill_constant(value=0.0, **locals())
