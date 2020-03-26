#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define functions to manipulate a tensor  
 __all__ = ['cast',
            'concat',
            'expand',
            'expand_as',
            'flatten',
            'gather',
            'gather_nd',
            'reshape',
            'reverse',
            'scatter',
            'scatter_nd_add',
            'scatter_nd',
            'shard_index',
            'slice',
            'split',
            'squeeze',
            'stack',
            'strided_slice',
            'unique',
            'unique_with_counts',
            'unsqueeze',
            'unstack',
            'flip',
            'unbind',
            'roll']


def stack(x, axis=0):
    """

    This OP stacks all the inputs :code:`x` along axis.

    .. code-block:: text

        Case 1:

          Input:
            x[0].shape = [1, 2]
            x[0].data = [ [1.0 , 2.0 ] ]
            x[1].shape = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[2].shape = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]

          Attrs:
            axis = 0

          Output:
            Out.dims = [3, 1, 2]
            Out.data =[ [ [1.0, 2.0] ],
                        [ [3.0, 4.0] ],
                        [ [5.0, 6.0] ] ]


        Case 2:


          Input:
            x[0].shape = [1, 2]
            x[0].data = [ [1.0 , 2.0 ] ]
            x[1].shape = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[2].shape = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]


          Attrs:
            axis = 1 or axis = -2

          Output:
            Out.shape = [1, 3, 2]
            Out.data =[ [ [1.0, 2.0]
                          [3.0, 4.0]
                          [5.0, 6.0] ] ]


    Args:
        x (Variable|list(Variable)): Input :code:`x` can be a single Tensor, a :code:`list` of Tensors.
                                     If :code:`x` is a :code:`list`, the shapes of all these Tensors
                                     must be the same. Supposing input is N dims
                                     Tensors :math:`[d_0, d_1, ..., d_{n-1}]`, the output is N+1 dims
                                     Tensor :math:`[d_0, d_1, d_{axis-1}, len(x), d_{axis}, ..., d_{n-1}]`.
                                     Support data types: float32, float64, int32, int64.
        axis (int, optional): The axis along which all inputs are stacked. ``axis`` range is :math:`[-(R+1), R+1)`.
                              R is the first tensor of inputs. If ``axis`` < 0, :math:`axis=axis+rank(x[0])+1`.
                              The default value of axis is 0.

    Returns:
        Variable: The stacked Tensor, has same data type with input Tensors. Output dim is :math:`rank(x[0])+1`.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            import paddle.tensor as tensor
            # set batch size=None
            x1 = fluid.data(name='x1', shape=[None, 1, 2], dtype='int32')
            x2 = fluid.data(name='x2', shape=[None, 1, 2], dtype='int32')
            # stack Tensor list
            data = tensor.stack([x1,x2]) # stack according to axis 0, data.shape=[2, None, 1, 2]

            data = tensor.stack([x1,x2], axis=1) # stack according to axis 1, data.shape=[None, 2, 1, 2]

            # stack single Tensor
            data = tensor.stack(x1)  # stack according to axis 0, data.shape=[1, None, 1, 2]

    """

    helper = LayerHelper('stack', **locals())
    axis = 0 if axis is None else axis

    if not isinstance(x, list) and not isinstance(x, tuple):
        x = [x]
    out = helper.create_variable_for_type_inference(x[0].dtype)
    if not in_dygraph_mode() and \
            x[0].desc.type() == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        assert len(x) == 1, "If the elements of 'x' in stack are Variable(LoDTensorArray), " \
                            "number of the elements must be 1, but received %s." % len(x)
        out_index = helper.create_variable_for_type_inference(dtype="int32")
        helper.append_op(
            type='tensor_array_to_tensor',
            inputs={'X': x[0]},
            outputs={'Out': [out],
                     'OutIndex': [out_index]},
            attrs={'axis': axis,
                   'use_stack': True})
    else:
        helper.append_op(
            type='stack',
            inputs={'X': x},
            outputs={'Y': out},
            attrs={'axis': axis})

    return out


def split(input, num_or_sections, dim=-1, name=None):
    """
    Split the input tensor into multiple sub-Tensors.

    Args:
        input (Variable): The input variable which is an N-D Tensor or LoDTensor, data type being float32, float64, int32 or int64.
        num_or_sections (int|list|tuple): If :attr:`num_or_sections` is an integer,
            then the integer indicates the number of equal sized sub-Tensors
            that the Tensor will be divided into. If :attr:`num_or_sections`
            is a list or tuple, the length of it indicates the number of
            sub-Tensors and the elements in it indicate the sizes of sub-Tensors'
            :attr:`dim` dimension orderly. The length of the list mustn't be larger than the Tensor's size of :attr:`dim` .
        dim (int32|Varible, optional): A scalar with type ``int32`` or a ``Tensor`` with shape [1] and type ``int32``. The dimension along which to split. If :math:`dim < 0`, the
            dimension to split along is :math:`rank(input) + dim`. Default is -1.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        list(Variable): The list of segmented Tensor variables.

    Raises:
        TypeError: num_or_sections is not int, list or tuple.
        TypeError: dim is not int or Variable.

    Example:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.tensor as tensor

            # input is a variable which shape is [3, 9, 5]
            input = fluid.data(
                 name="input", shape=[3, 9, 5], dtype="float32")

            x0, x1, x2 = tensor.split(input, num_or_sections=3, dim=1)
            # x0.shape [3, 3, 5]
            # x1.shape [3, 3, 5]
            # x2.shape [3, 3, 5]

            x0, x1, x2 = tensor.split(input, num_or_sections=[2, 3, 4], dim=1)
            # x0.shape [3, 2, 5]
            # x1.shape [3, 3, 5]
            # x2.shape [3, 4, 5]

            x0, x1, x2 = tensor.split(input, num_or_sections=[2, 3, -1], dim=1)
            # x0.shape [3, 2, 5]
            # x1.shape [3, 3, 5]
            # x2.shape [3, 4, 5]
    """
    if in_dygraph_mode():
        inputs = {'X': [input]}
        attrs = {}
        if isinstance(dim, Variable):
            dim = dim.numpy()
            assert dim.shape == (1,
                                 ), "dim of type Variable should have shape [1]"
            dim = dim[0]
        dim = (len(input.shape) + dim) if dim < 0 else dim
        attrs['axis'] = dim

        if isinstance(num_or_sections, int):
            num = num_or_sections
            attrs['num'] = num_or_sections
        elif isinstance(num_or_sections, (list, tuple)):
            num = len(num_or_sections)
            if utils._contain_var(num_or_sections):
                raise TypeError(
                    "The type of 'num_or_sections' in split must be int or list[int] or tuple[int] in Dygraph mode, but "
                    "received %s, which contains Variable." %
                    (type(num_or_sections)))
            else:
                attrs['sections'] = list(num_or_sections)
        else:
            raise TypeError(
                "The type of 'num_or_sections' in split must be int or list in Dygraph mode, but "
                "received %s." % (type(num_or_sections)))

        res = core.ops.split(inputs, attrs, {}, {'Out': num})
        return res['Out']

    if not isinstance(num_or_sections, (int, list, tuple)):
        raise TypeError(
            "The type of 'num_or_sections' in split must be int, list or "
            "tuple, but received %s." % (type(num_or_sections)))
    if not isinstance(dim, (int, Variable)):
        raise TypeError(
            "The type of 'dim' in split must be int or Variable, but "
            "received %s." % (type(dim)))

    helper = LayerHelper('split', **locals())
    input_shape = input.shape
    inputs = {'X': input}
    attrs = {'num': num_or_sections if isinstance(num_or_sections, int) else 0}

    def _get_SectionsTensorList(one_list):
        tensor_list = []
        unk_dim_idx = -1
        for idx, dim_size in enumerate(one_list):
            if isinstance(dim_size, Variable):
                dim_size.stop_gradient = True
                tensor_list.append(dim_size)
            else:
                assert (isinstance(dim_size, int))
                if dim_size == -1:
                    assert unk_dim_idx == -1, (
                        "Only one value of 'num_or_section' in split can "
                        "be -1. But received num_or_section[%d] is also -1." %
                        idx)
                    unk_dim_idx = idx
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant(
                    [1], 'int32', dim_size, force_cpu=True, out=temp_out)
                tensor_list.append(temp_out)
        return tensor_list

    if isinstance(dim, Variable):
        dim.stop_gradient = True
        inputs['AxisTensor'] = dim
    else:
        dim = (len(input_shape) + dim) if dim < 0 else dim
        attrs['axis'] = dim

    if isinstance(num_or_sections, int):
        assert num_or_sections > 1, 'num_or_sections must be more than 1.'
        if isinstance(dim, int) and input_shape[dim] > 0:
            assert input_shape[dim] % num_or_sections ==0, \
                "The input's size along the split dimension " \
                "must be evenly divisible by Attr(num_or_sections). " \
                "But %d is not evenly divisible by %d. " % (num_or_sections,input_shape[dim])
        num = num_or_sections
    else:
        if isinstance(dim, int) and input_shape[dim] > 0:
            assert len(num_or_sections) <= input_shape[
                dim], 'len(num_or_sections) must not be more than input.shape[dim].'
        num = len(num_or_sections)
        attrs['sections'] = list(
            map(lambda ele: -1 if isinstance(ele, Variable) else ele,
                num_or_sections))
        if utils._contain_var(num_or_sections):
            inputs['SectionsTensorList'] = _get_SectionsTensorList(
                num_or_sections)

    outs = [
        helper.create_variable_for_type_inference(dtype=helper.input_dtype())
        for i in range(num)
    ]
    helper.append_op(
        type='split', inputs=inputs, outputs={'Out': outs}, attrs=attrs)
    return outs


def squeeze(input, axes, name=None):
    """
    This OP will squeeze single-dimensional entries of input tensor's shape. If axes is provided, will
    remove the dims by axes, the dims selected by axes should be one. If not provide axes, all dims equal
    to one will be deleted.


    .. code-block:: text 

        Case1:

          Input:
            X.shape = (1, 3, 1, 5)
            axes = [0]
          Output:
            Out.shape = (3, 1, 5)

        Case2:

          Input:
            X.shape = (1, 3, 1, 5)
            axes = []
          Output:
            Out.shape = (3, 5)

        Case3:

          Input:
            X.shape = [1,3,1,5]
            axes = [-2]
          Output:
            Out.shape = [1,3,5]

    Args:
        input (Variable): The input Tensor. Support data type: float32, float64, int8, int32, int64.
                          axes (list): One integer or List of integers, indicating the dimensions to be squeezed.
                          Axes range is :math:`[-rank(input), rank(input))`.
                          If axes is negative, :math:`axes=axes+rank(input)`.
        name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.

    Returns:
        Variable: Output squeezed Tensor. Data type is same as input Tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            import paddle.tensor as tensor
            # set batch size=None
            x = fluid.data(name='x', shape=[None, 5, 1, 10])
            y = tensor.squeeze(input=x, axes=[2]) # y.shape=[None, 5, 10]

    """
    helper = LayerHelper("squeeze", **locals())
    check_variable_and_dtype(input, 'input',
                             ['float32', 'float64', 'int8', 'int32', 'int64'],
                             'squeeze')
    check_type(axes, 'axes', list, 'squeeze')
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="squeeze2",
        inputs={"X": input},
        attrs={"axes": axes},
        outputs={"Out": out,
                 "XShape": x_shape})

    return out


def unsqueeze(input, axes, name=None):
    """
    Insert single-dimensional entries to the shape of a Tensor. Takes one
    required argument axes, a list of dimensions that will be inserted.
    Dimension indices in axes are as seen in the output tensor.

    For example:

    .. code-block:: text

      Given a tensor such that tensor with shape [3, 4, 5],
      then Unsqueezed tensor with axes=[0, 4] has shape [1, 3, 4, 5, 1].

    Args:
        input (Variable): The input Tensor to be unsqueezed. It is a N-D Tensor of data types float32, float64, int32.
        axes (int|list|tuple|Variable): Indicates the dimensions to be inserted. The data type is ``int32`` . If ``axes`` is a list or tuple, the elements of it should be integers or Tensors with shape [1]. If ``axes`` is an Variable, it should be an 1-D Tensor .
        name (str|None): Name for this layer.

    Returns:
        Variable: Output unsqueezed Tensor, with data type being float32, float64, int32, int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.tensor as tensor
            x = fluid.layers.data(name='x', shape=[5, 10])
            y = tensor.unsqueeze(input=x, axes=[1])

    """
    if not isinstance(axes, (int, list, tuple, Variable)):
        raise TypeError(
            "The type of 'axes' in unsqueeze must be int, list, tuple or Variable, but "
            "received %s." % (type(axes)))
    helper = LayerHelper("unsqueeze2", **locals())
    inputs = {"X": input}
    attrs = {}

    def _to_Variable_list(one_list):
        Variable_list = []
        for ele in one_list:
            if isinstance(ele, Variable):
                ele.stop_gradient = True
                Variable_list.append(ele)
            else:
                assert (isinstance(ele, int))
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', ele, force_cpu=True, out=temp_out)
                Variable_list.append(temp_out)
        return Variable_list

    if isinstance(axes, int):
        axes = [axes]
    if isinstance(axes, Variable):
        axes.stop_gradient = True
        inputs["AxesTensor"] = axes
    elif isinstance(axes, (list, tuple)):
        if utils._contain_var(axes):
            inputs["AxesTensorList"] = _to_Variable_list(axes)
        else:
            attrs["axes"] = axes

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="unsqueeze2",
        inputs=inputs,
        attrs=attrs,
        outputs={"Out": out,
                 "XShape": x_shape})

    return out


def gather(input, index, overwrite=True):
    """
    **Gather Layer**

    Output is obtained by gathering entries of the outer-most dimension
    of X indexed by `index` and concatenate them together.

    .. math::

        Out = X[Index]


    .. code-block:: text


                Given:

                X = [[1, 2],
                     [3, 4],
                     [5, 6]]

                Index = [1, 2]

                Then:

                Out = [[3, 4],
                       [5, 6]]

    Args:
        input (Variable): The source input tensor with rank>=1. Supported data type is 
            int32, int64, float32, float64 and uint8 (only for CPU), 
            float16 (only for GPU).
        index (Variable): The index input tensor with rank=1. Data type is int32 or int64.
        overwrite (bool, optional): The mode that updating the grad when has same index.
            If True, use the overwrite mode to update the grad of the same index,
	    if False, use the accumulate mode to update the grad of the same index. 
	    Default value is True.
	    


    Returns:
        output (Variable): The output is a tensor with the same rank as input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.tensor as tensor
            x = fluid.data(name='x', shape=[-1, 5], dtype='float32')
            index = fluid.data(name='index', shape=[-1, 1], dtype='int32')
            output = tensor.gather(x, index)
    """
    helper = LayerHelper('gather', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="gather",
        inputs={"X": input,
                "Index": index},
        outputs={"Out": out},
        attrs={'overwrite': overwrite})
    return out
