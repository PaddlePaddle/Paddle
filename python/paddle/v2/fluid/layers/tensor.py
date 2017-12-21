from ..layer_helper import LayerHelper

__all__ = [
    'create_tensor', 'cast', 'concat', 'sums', 'assign',
    'fill_constant_batch_size_like', 'fill_constant', 'ones', 'zeros'
]


def create_tensor(dtype, name=None):
    helper = LayerHelper("create_tensor", **locals())
    return helper.create_variable(name=helper.name, dtype=dtype)


def cast(x, dtype):
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


def concat(input, axis):
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


def sums(input, out=None):
    """
    This function takes in the input and performs the sum operation on it
    and returns that as the output.
    """
    helper = LayerHelper('sum', **locals())
    if out is None:
        out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(type='sum', inputs={'X': input}, outputs={'Out': out})
    return out


def assign(input, output):
    helper = LayerHelper('assign', **locals())
    helper.append_op(
        type='scale',
        inputs={'X': [input]},
        outputs={'Out': [output]},
        attrs={'scale': 1.0})
    return output


def fill_constant(shape, dtype, value, out=None):
    """
    **fill_constant**

    This function creates a tensor of specified *shape* and 
    *dtype*, and initializes this with a constant supplied in *value*.
    
    It also sets *stop_gradient* to True.

    Args:
        shape(tuple|list|None): Shape of output tensor
        dtype(np.dtype|core.DataType|str): Data type of output tensor
        value(float): Constant value to initialize the output tensor
        out(Variable): Output Variable to initialize

    Returns:
        Variable: The tensor variable storing the output

    Examples:
        .. code-block:: python

          data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')
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
                                  output_dim_idx=0):
    """
    **fill_constant_batch_size_like**

    This function creates a tensor of specified *shape*, *dtype* and batch size, 
    and initializes this with a constant supplied in *value*. The batch size is 
    obtained from the `input` tensor. 

    It also sets *stop_gradient* to True.

    Args:
        input(Variable): Tensor whose dimensions will be used to get batch size
        shape(tuple|list|None): Shape of output tensor
        dtype(np.dtype|core.DataType|str): Data type of output tensor
        value(float): Constant value to initialize the output tensor
        input_dim_idx(int): Index of input's batch size dimension
        output_dim_idx(int): Index of output's batch size dimension

    Returns:
        Variable: The tensor variable storing the output

    Examples:
        .. code-block:: python

          data = fluid.layers.fill_constant(shape=[1], value=0, dtype='int64')
    """
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


def ones(shape, dtype):
    """
    This function performs the same function as fill_constant() declared above
    with the constant value being 1.0.
    """
    return fill_constant(value=1.0, **locals())


def zeros(shape, dtype):
    """
    This function performs the same function as fill_constant() declared above
    with the constant value being 0.0.
    """
    return fill_constant(value=0.0, **locals())
