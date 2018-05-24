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


def concat(input, axis=0):
    """
    **Concat**

    This function concatenates the input along the axis mentioned
    and returns that as the output.

    Args:
        input(list): List of tensors to be concatenated
        axis(int): Integer axis along which the tensors will be concatenated

    Returns:
        Variable: Output variable of the concatenation

    Examples:
        .. code-block:: python
          out = fluid.layers.concat(input=[Efirst, Esecond, Ethird, Efourth])
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
    """This function performs the sum operation on the input and returns the
    result as the output.

    Args:
        input (Variable|list): The input tensor that has the elements
                               that need to be summed up.

    Returns:
        Variable: The tensor type variable that has the sum of input
                  written to it.

    Examples:
        .. code-block::python

          tmp = fluid.layers.zeros(shape=[10], dtype='int32')
          i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
          a0 = layers.array_read(array=tmp, i=i)
          i = layers.increment(x=i)
          a1 = layers.array_read(array=tmp, i=i)
          mean_a0 = layers.mean(x=a0)
          mean_a1 = layers.mean(x=a1)
          a_sum = layers.sums(input=[mean_a0, mean_a1])
    """
    helper = LayerHelper('sum', **locals())
    if out is None:
        out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(type='sum', inputs={'X': input}, outputs={'Out': out})
    return out


def assign(input, output):
    """
    **Assign**

    This function copies the *input* Variable to the *output* Variable.

    Args:
        input(Variable): The source variable
        output(Variable): The destination variable

    Returns:
        Variable: The destination variable that was supplied as the *output*.

    Examples:
        .. code-block:: python
          out = fluid.layers.create_tensor(dtype='float32')
          hidden = fluid.layers.fc(input=data, size=10)
          fluid.layers.assign(hidden, out)
    """
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
