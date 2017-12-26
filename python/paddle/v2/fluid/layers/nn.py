"""
All layers just related to the neural network.
"""

from ..layer_helper import LayerHelper
from ..initializer import Normal, Constant
from ..framework import Variable
from ..param_attr import ParamAttr
from tensor import concat

__all__ = [
    'fc', 'embedding', 'dynamic_lstm', 'gru_unit', 'linear_chain_crf',
    'crf_decoding', 'cos_sim', 'cross_entropy', 'square_error_cost', 'accuracy',
    'chunk_eval', 'sequence_conv', 'conv2d', 'sequence_pool', 'pool2d',
    'batch_norm', 'beam_search_decode', 'conv2d_transpose', 'sequence_expand',
    'lstm_unit', 'reduce_sum', 'reduce_mean', 'sequence_first_step',
    'sequence_last_step'
]


def fc(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       act=None,
       name=None):
    """
    **Fully Connected Layer**

    The fully connected layer can take multiple tensors as its inputs. It
    creates a variable (one for each input tensor) called weights for each input
    tensor, which represents a fully connected weight matrix from each input
    unit to each output unit. The fully connected layer multiplies each input
    tensor with its coresponding weight to produce an output Tensor. If
    multiple input tensors are given, the results of multiple multiplications
    will be sumed up. If bias_attr is not None, a biases variable will be
    created and added to the output. Finally, if activation is not None,
    it will be applied to the output as well.

    This process can be formulated as follows:

    .. math::

        Out = Act({\sum_{i=0}^{N-1}W_iX_i + b})

    In the above equation:

    * :math:`N`: Number of the input.
    * :math:`X_i`: The input tensor.
    * :math:`W`: The weights created by this layer.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation funtion.
    * :math:`Out`: The output tensor.

    Args:
       input(Variable|list): The input tensor(s) to the fully connected layer.
       size(int): The number of output units in the fully connected layer.
       num_flatten_dims(int): The fc layer can accept an input tensor with more
                              than two dimensions. If this happens, the
                              multidimensional tensor will first be flattened
                              into a 2-dimensional matrix. The parameter
                              `num_flatten_dims` determines how the input tensor
                              is flattened: the first `num_flatten_dims`
                              dimensions will be flatten to form the first
                              dimension of the final matrix (height of the
                              matrix), and the rest `rank(X) - num_col_dims`
                              dimensions are flattened to form the second
                              dimension of the final matrix (width of the matrix).
                              For example, suppose `X` is a 6-dimensional tensor
                              with a shape [2, 3, 4, 5, 6], and
                              `x_num_col_dims` = 3. Then, the flattened matrix
                              will have a shape [2 x 3 x 4, 5 x 6] = [24, 30].
                              By default, `x_num_col_dims` is set to 1.
       param_attr(ParamAttr|list): The parameter attribute for learnable
                                   parameters/weights of the fully connected
                                   layer.
       param_initializer(ParamAttr|list): The initializer used for the
                                          weight/parameter. If set None,
                                          XavierInitializer() will be used.
       bias_attr(ParamAttr|list): The parameter attribute for the bias parameter
                                  for this layer. If set None, no bias will be
                                  added to the output units.
       bias_initializer(ParamAttr|list): The initializer used for the bias.
                                        If set None, then ConstantInitializer()
                                        will be used.
       act(str): Activation to be applied to the output of the fully connected
                 layer.
       name(str): Name/alias of the fully connected layer.


    Returns:
        Variable: The output tensor variable.

    Raises:
        ValueError: If rank of the input tensor is less than 2.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
          fc = fluid.layers.fc(input=data, size=1000, act="tanh")
    """

    helper = LayerHelper("fc", **locals())

    dtype = helper.input_dtype()

    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1)
        ] + [size]
        w = helper.create_parameter(
            attr=param_attr, shape=param_shape, dtype=dtype, is_bias=False)
        tmp = helper.create_tmp_variable(dtype)
        helper.append_op(
            type="mul",
            inputs={
                "X": input_var,
                "Y": w,
            },
            outputs={"Out": tmp},
            attrs={"x_num_col_dims": num_flatten_dims,
                   "y_num_col_dims": 1})
        mul_results.append(tmp)

    # sum
    if len(mul_results) == 1:
        pre_bias = mul_results[0]
    else:
        pre_bias = helper.create_tmp_variable(dtype)
        helper.append_op(
            type="sum", inputs={"X": mul_results}, outputs={"Out": pre_bias})
    # add bias
    pre_activation = helper.append_bias_op(pre_bias)
    # add activation
    return helper.append_activation(pre_activation)


def embedding(input, size, is_sparse=False, param_attr=None, dtype='float32'):
    """
    **Embedding Layer**

    This layer is used to lookup a vector of IDs, provided by *input*, in a lookup table.
    The result of this lookup is the embedding of each ID in the *input*.

    All the input variables are passed in as local variables to the LayerHelper
    constructor.

    Args:
       input(Variable): Input to the function
       size(tuple|list|None): Shape of the look up table parameter 
       is_sparse(bool): Boolean flag that specifying whether the input is sparse
       param_attr(ParamAttr): Parameters for this layer
       dtype(np.dtype|core.DataType|str): The type of data : float32, float_16, int etc

    Returns:
        Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          dict_size = len(dataset.ids)
          data = fluid.layers.data(name='ids', shape=[32, 32], dtype='float32')
          fc = fluid.layers.embedding(input=data, size=[dict_size, 16])
    """

    helper = LayerHelper('embedding', **locals())
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False)
    tmp = helper.create_tmp_variable(dtype)
    helper.append_op(
        type='lookup_table',
        inputs={'Ids': input,
                'W': w},
        outputs={'Out': tmp},
        attrs={'is_sparse': is_sparse})
    return tmp


# TODO(qijun): expose H0 and C0
def dynamic_lstm(input,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 use_peepholes=True,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 cell_activation='tanh',
                 candidate_activation='tanh',
                 dtype='float32'):
    helper = LayerHelper('lstm', **locals())
    size = size / 4
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 4 * size], dtype=dtype)
    bias_size = [1, 7 * size]
    if not use_peepholes:
        bias_size[1] = 4 * size
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

    hidden = helper.create_tmp_variable(dtype)
    cell = helper.create_tmp_variable(dtype)
    batch_gate = helper.create_tmp_variable(dtype)
    batch_cell_pre_act = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='lstm',
        inputs={'Input': input,
                'Weight': weight,
                'Bias': bias},
        outputs={
            'Hidden': hidden,
            'Cell': cell,
            'BatchGate': batch_gate,
            'BatchCellPreAct': batch_cell_pre_act
        },
        attrs={
            'use_peepholes': use_peepholes,
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'cell_activation': cell_activation,
            'candidate_activation': candidate_activation
        })
    return hidden, cell


def gru_unit(input,
             hidden,
             size,
             weight=None,
             bias=None,
             activation='tanh',
             gate_activation='sigmoid'):
    """
    GRUUnit Operator implements partial calculations of the GRU unit as following:

    $$
    update \ gate: u_t = actGate(xu_t + W_u * h_{t-1} + b_u) \\
    reset \ gate: r_t = actGate(xr_t + W_r * h_{t-1} + b_r)  \\
    output \ candidate: {h}_t = actNode(xc_t + W_c * dot(r_t, h_{t-1}) + b_c) \\
    output: h_t = dot((1 - u_t), h_{t-1}) + dot(u_t, {h}_t)
    $$

    which is same as one time step of GRU Operator.

    @note To implement the complete GRU unit, fully-connected operator must be
    used before to feed xu, xr and xc as the Input of GRUUnit operator.

    TODO(ChunweiYan) add more document here
    """
    activation_dict = dict(
        identity=0,
        sigmoid=1,
        tanh=2,
        relu=3, )
    activation = activation_dict[activation]
    gate_activation = activation_dict[gate_activation]

    helper = LayerHelper('gru_unit', **locals())
    dtype = helper.input_dtype()
    size = size / 3

    # create weight
    if weight is None:
        weight = helper.create_parameter(
            attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype)

    # create bias
    if bias is None:
        bias_size = [1, 3 * size]
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

    gate = helper.create_tmp_variable(dtype)
    reset_hidden_pre = helper.create_tmp_variable(dtype)
    updated_hidden = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='gru_unit',
        inputs={'Input': input,
                'HiddenPrev': hidden,
                'Weight': weight},
        outputs={
            'Gate': gate,
            'ResetHiddenPrev': reset_hidden_pre,
            'Hidden': updated_hidden,
        },
        attrs={
            'activation': 0,
            'gate_activation': 1,
        })

    return updated_hidden, reset_hidden_pre, gate


def linear_chain_crf(input, label, param_attr=None):
    helper = LayerHelper('linear_chain_crf', **locals())
    size = input.shape[1]
    transition = helper.create_parameter(
        attr=helper.param_attr,
        shape=[size + 2, size],
        dtype=helper.input_dtype())
    alpha = helper.create_tmp_variable(dtype=helper.input_dtype())
    emission_exps = helper.create_tmp_variable(dtype=helper.input_dtype())
    transition_exps = helper.create_tmp_variable(dtype=helper.input_dtype())
    log_likelihood = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='linear_chain_crf',
        inputs={"Emission": [input],
                "Transition": transition,
                "Label": label},
        outputs={
            "Alpha": [alpha],
            "EmissionExps": [emission_exps],
            "TransitionExps": transition_exps,
            "LogLikelihood": log_likelihood
        })

    return log_likelihood


def crf_decoding(input, param_attr, label=None):
    helper = LayerHelper('crf_decoding', **locals())
    transition = helper.get_parameter(param_attr.name)
    viterbi_path = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='crf_decoding',
        inputs={"Emission": [input],
                "Transition": transition,
                "Label": label},
        outputs={"ViterbiPath": [viterbi_path]})

    return viterbi_path


def cos_sim(X, Y, **kwargs):
    """
    This function performs the cosine similarity between two tensors
    X and Y and returns that as the output.
    """
    helper = LayerHelper('cos_sim', **kwargs)
    out = helper.create_tmp_variable(dtype=X.dtype)
    xnorm = helper.create_tmp_variable(dtype=X.dtype)
    ynorm = helper.create_tmp_variable(dtype=X.dtype)
    helper.append_op(
        type='cos_sim',
        inputs={'X': [X],
                'Y': [Y]},
        outputs={'Out': [out],
                 'XNorm': [xnorm],
                 'YNorm': [ynorm]})
    return out


def cross_entropy(input, label, **kwargs):
    """
    This function computes cross_entropy using the input and label.
    """
    helper = LayerHelper('cross_entropy', **kwargs)
    out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]},
        attrs=kwargs)
    return out


def square_error_cost(input, label, **kwargs):
    """
    This functions returns the squared error cost using the input and label.
    The output is appending the op to do the above.
    """
    helper = LayerHelper('square_error_cost', **kwargs)
    minus_out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='elementwise_sub',
        inputs={'X': [input],
                'Y': [label]},
        outputs={'Out': [minus_out]})

    square_out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='square', inputs={'X': [minus_out]}, outputs={'Y': [square_out]})
    return square_out


def accuracy(input, label, k=1, correct=None, total=None, **kwargs):
    """
    This function computes the accuracy using the input and label.
    The output is the top_k inputs and their indices.
    """
    helper = LayerHelper("accuracy", **kwargs)
    topk_out = helper.create_tmp_variable(dtype=input.dtype)
    topk_indices = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="top_k",
        inputs={"X": [input]},
        outputs={"Out": [topk_out],
                 "Indices": [topk_indices]},
        attrs={"k": k})
    acc_out = helper.create_tmp_variable(dtype="float32")
    if correct is None:
        correct = helper.create_tmp_variable(dtype="int64")
    if total is None:
        total = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="accuracy",
        inputs={
            "Out": [topk_out],
            "Indices": [topk_indices],
            "Label": [label]
        },
        outputs={
            "Accuracy": [acc_out],
            "Correct": [correct],
            "Total": [total],
        })
    return acc_out


def chunk_eval(input,
               label,
               chunk_scheme,
               num_chunk_types,
               excluded_chunk_types=None,
               **kwargs):
    """
    This function computes and outputs the precision, recall and
    F1-score of chunk detection.
    """
    helper = LayerHelper("chunk_eval", **kwargs)

    # prepare output
    precision = helper.create_tmp_variable(dtype="float32")
    recall = helper.create_tmp_variable(dtype="float32")
    f1_score = helper.create_tmp_variable(dtype="float32")
    num_infer_chunks = helper.create_tmp_variable(dtype="int64")
    num_label_chunks = helper.create_tmp_variable(dtype="int64")
    num_correct_chunks = helper.create_tmp_variable(dtype="int64")

    helper.append_op(
        type="chunk_eval",
        inputs={"Inference": [input],
                "Label": [label]},
        outputs={
            "Precision": [precision],
            "Recall": [recall],
            "F1-Score": [f1_score],
            "NumInferChunks": [num_infer_chunks],
            "NumLabelChunks": [num_label_chunks],
            "NumCorrectChunks": [num_correct_chunks]
        },
        attrs={
            "num_chunk_types": num_chunk_types,
            "chunk_scheme": chunk_scheme,
            "excluded_chunk_types": excluded_chunk_types or []
        })
    return precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks


def sequence_conv(input,
                  num_filters,
                  filter_size=3,
                  filter_stride=1,
                  padding=None,
                  bias_attr=None,
                  param_attr=None,
                  act=None):
    """
    This function creates the op for sequence_conv, using the inputs and
    other convolutional configurations for the filters and stride as given
    in the input parameters to the function.
    """

    # FIXME(dzh) : want to unify the argument of python layer
    # function. So we ignore some unecessary attributes.
    # such as, padding_trainable, context_start.

    helper = LayerHelper('sequence_conv', **locals())
    dtype = helper.input_dtype()
    filter_shape = [filter_size * input.shape[1], num_filters]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    pre_bias = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='sequence_conv',
        inputs={
            'X': [input],
            'Filter': [filter_param],
        },
        outputs={"Out": pre_bias},
        attrs={
            'contextStride': filter_stride,
            'contextStart': -int(filter_size / 2),
            'contextLength': filter_size
        })
    pre_act = helper.append_bias_op(pre_bias)
    return helper.append_activation(pre_act)


def conv2d(input,
           num_filters,
           filter_size,
           stride=None,
           padding=None,
           groups=None,
           param_attr=None,
           bias_attr=None,
           act=None,
           name=None):
    """
    This function creates the op for a 2-dimensional Convolution.
    This is performed using the parameters of filters(size, dimensionality etc)
    , stride and other configurations for a Convolution operation.
    This funciton can also append an activation on top of the
    conv-2d output, if mentioned in the input parameters.
    """

    if stride is None:
        stride = [1, 1]
    helper = LayerHelper('conv2d', **locals())
    dtype = helper.input_dtype()

    num_channels = input.shape[1]
    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels / groups

    if isinstance(filter_size, int):
        filter_size = [filter_size, filter_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(padding, int):
        padding = [padding, padding]

    input_shape = input.shape
    filter_shape = [num_filters, num_filter_channels] + filter_size

    def _get_default_param_initializer():
        std = (2.0 / (filter_size[0]**2 * num_channels))**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='conv2d_cudnn',
        inputs={
            'Input': input,
            'Filter': filter_param,
        },
        outputs={"Output": pre_bias},
        attrs={'strides': stride,
               'paddings': padding,
               'groups': groups})

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)

    return helper.append_activation(pre_act)


def sequence_pool(input, pool_type, **kwargs):
    """
    This function add the operator for sequence pooling. 
    It pools features of all time-steps of each instance, and is applied 
    on top of the input using pool_type mentioned in the parameters. 

    It supports four pool_type:

    - average: :math:`Out[i] = \\frac{\sum_i X_i}{N}`
    - sum:     :math:`Out[i] = \sum_jX_{ij}`
    - sqrt:    :math:`Out[i] = \\frac{\sum_jX_{ij}}{\sqrt{len(X_i)}}`
    - max:     :math:`Out[i] = max(X_i)`

    .. code-block:: text

       x is a 1-level LoDTensor:
         x.lod = [[0, 2, 5, 7]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [3, 1]
         with condition len(x.lod[-1]) - 1 == out.dims[0]

       for different pool_type:
         average: out.data = [2, 4, 3], where 2=(1+3)/2, 4=(2+4+6)/3, 3=(5+1)/2
         sum    : out.data = [4, 12, 6], where 4=1+3, 12=2+4+6, 6=5+1
         sqrt   : out.data = [2.82, 6.93, 4.24], where 2.82=(1+3)/sqrt(2),
                    6.93=(2+4+6)/sqrt(3), 4.24=(5+1)/sqrt(2)
         max    : out.data = [3, 6, 5], where 3=max(1,3), 6=max(2,4,6), 5=max(5,1)
         
    Args:
        input(variable): The input variable which is a LoDTensor.
        pool_type (string): The pooling type of sequence_pool. 
            It supports average, sum, sqrt and max.

    Returns:
        The sequence pooling variable which is a Tensor.

    Examples:

        .. code-block:: python
             
             x = fluid.layers.data(name='x', shape=[7, 1], 
                              dtype='float32', lod_level=1)
             avg_x = fluid.layers.sequence_pool(input=x, pool_type='average')
             sum_x = fluid.layers.sequence_pool(input=x, pool_type='sum')
             sqrt_x = fluid.layers.sequence_pool(input=x, pool_type='sqrt')
             max_x = fluid.layers.sequence_pool(input=x, pool_type='max')
    """
    helper = LayerHelper('sequence_pool', input=input, **kwargs)
    dtype = helper.input_dtype()
    pool_out = helper.create_tmp_variable(dtype)
    max_index = helper.create_tmp_variable(dtype)

    helper.append_op(
        type="sequence_pool",
        inputs={"X": input},
        outputs={"Out": pool_out,
                 "MaxIndex": max_index},
        attrs={"pooltype": pool_type.upper()})

    return pool_out


def sequence_first_step(input, **kwargs):
    """
    This funciton get the first step of sequence.

    .. code-block:: text

       x is a 1-level LoDTensor:
         x.lod = [[0, 2, 5, 7]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [3, 1]
         with condition len(x.lod[-1]) - 1 == out.dims[0]
         out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)
         
    Args:
        input(variable): The input variable which is a LoDTensor.

    Returns:
        The sequence's first step variable which is a Tensor.

    Examples:

        .. code-block:: python
             
             x = fluid.layers.data(name='x', shape=[7, 1], 
                              dtype='float32', lod_level=1)
             x_first_step = fluid.layers.sequence_first_step(input=x)
    """
    return sequence_pool(input=input, pool_type="first")


def sequence_last_step(input, **kwargs):
    """
    This funciton get the last step of sequence.

    .. code-block:: text

       x is a 1-level LoDTensor:
         x.lod = [[0, 2, 5, 7]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [3, 1]
         with condition len(x.lod[-1]) - 1 == out.dims[0]
         out.data = [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)
         
    Args:
        input(variable): The input variable which is a LoDTensor.

    Returns:
        The sequence's last step variable which is a Tensor.

    Examples:

        .. code-block:: python
             
             x = fluid.layers.data(name='x', shape=[7, 1], 
                              dtype='float32', lod_level=1)
             x_last_step = fluid.layers.sequence_last_step(input=x)
    """
    return sequence_pool(input=input, pool_type="last")


def pool2d(input,
           pool_size,
           pool_type,
           pool_stride=None,
           pool_padding=None,
           global_pooling=False):
    """
    This function adds the operator for pooling in 2 dimensions, using the
    pooling configurations mentioned in input parameters.
    """
    if pool_padding is None:
        pool_padding = [0, 0]
    if pool_stride is None:
        pool_stride = [1, 1]
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))
    if isinstance(pool_size, int):
        pool_size = [pool_size, pool_size]
    if isinstance(pool_stride, int):
        pool_stride = [pool_stride, pool_stride]
    if isinstance(pool_padding, int):
        pool_padding = [pool_padding, pool_padding]

    helper = LayerHelper('pool2d', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_tmp_variable(dtype)

    helper.append_op(
        type="pool2d",
        inputs={"X": input},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding
        })

    return pool_out


def batch_norm(input,
               act=None,
               is_test=False,
               momentum=0.9,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               data_layout='NCHW'):
    """
    This function helps create an operator to implement
    the BatchNorm layer using the configurations from the input parameters.
    """
    helper = LayerHelper('batch_norm', **locals())
    dtype = helper.input_dtype()

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        default_initializer=Constant(1.0))

    bias = helper.create_parameter(
        attr=helper.param_attr, shape=param_shape, dtype=dtype, is_bias=True)

    mean = helper.create_global_variable(
        dtype=input.dtype, shape=param_shape, persistable=True)
    helper.set_variable_initializer(var=mean, initializer=Constant(0.0))

    variance = helper.create_global_variable(
        dtype=input.dtype, shape=param_shape, persistable=True)
    helper.set_variable_initializer(var=variance, initializer=Constant(1.0))

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance out share the same memory
    variance_out = variance
    saved_mean = helper.create_tmp_variable(dtype)
    saved_variance = helper.create_tmp_variable(dtype)

    batch_norm_out = helper.create_tmp_variable(dtype)

    helper.append_op(
        type="batch_norm",
        inputs={
            "X": input,
            "Scale": scale,
            "Bias": bias,
            "Mean": mean,
            "Variance": variance
        },
        outputs={
            "Y": batch_norm_out,
            "MeanOut": mean_out,
            "VarianceOut": variance_out,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance
        },
        attrs={"momentum": momentum,
               "epsilon": epsilon,
               "is_test": is_test})

    return helper.append_activation(batch_norm_out)


def beam_search_decode(ids, scores):
    helper = LayerHelper('beam_search_decode', **locals())
    sentence_ids = helper.create_tmp_variable(dtype=ids.dtype)
    sentence_scores = helper.create_tmp_variable(dtype=ids.dtype)

    helper.append_op(
        type="beam_search_decode",
        inputs={"Ids": ids,
                "Scores": scores},
        outputs={
            "SentenceIds": sentence_ids,
            "SentenceScores": sentence_scores
        })

    return sentence_ids, sentence_scores


def conv2d_transpose(input,
                     num_filters,
                     output_size=None,
                     filter_size=None,
                     padding=None,
                     stride=None,
                     dilation=None,
                     param_attr=None):
    """
    The transpose of conv2d layer.

    This layer is also known as deconvolution layer.

    Args:
        input(Variable): The input image with [N, C, H, W] format.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        output_size(int|tuple|None): The output image size. If output size is a
            tuple, it must contain two integers, (image_H, image_W). This
            parameter only works when filter_size is None.
        filter_size(int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.  None if use output size to
            calculate filter_size
        padding(int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding.
        stride(int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride.
        dilation(int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation.
        param_attr: Parameter Attribute.
        main_program(Program): the main program
        startup_program(Program): the startup program

    Returns:
        Variable: Output image.
    """
    helper = LayerHelper("conv2d_transpose", **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv2d_transpose must be Variable")
    input_channel = input.shape[1]

    op_attr = dict()

    if isinstance(padding, int):
        op_attr['paddings'] = [padding, padding]
    elif padding is not None:
        op_attr['paddings'] = padding

    if isinstance(stride, int):
        op_attr['strides'] = [stride, stride]
    elif stride is not None:
        op_attr['strides'] = stride

    if isinstance(dilation, int):
        op_attr['dilations'] = [dilation, dilation]
    elif dilation is not None:
        op_attr['dilations'] = dilation

    if filter_size is None:
        if output_size is None:
            raise ValueError("output_size must be set when filter_size is None")
        if isinstance(output_size, int):
            output_size = [output_size, output_size]

        padding = op_attr.get('paddings', [0, 0])
        stride = op_attr.get('strides', [1, 1])
        dilation = op_attr.get('dilations', [1, 1])

        h_in = input.shape[2]
        w_in = input.shape[3]

        filter_size_h = (output_size[0] - (h_in - 1) * stride[0] + 2 *
                         padding[0] - 1) / dilation[0] + 1
        filter_size_w = (output_size[1] - (w_in - 1) * stride[1] + 2 *
                         padding[1] - 1) / dilation[1] + 1
        filter_size = [filter_size_h, filter_size_w]

    elif isinstance(filter_size, int):
        filter_size = [filter_size, filter_size]

    filter_shape = [input_channel, num_filters] + filter_size
    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr)

    out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='conv2d_transpose',
        inputs={'Input': [input],
                'Filter': [img_filter]},
        outputs={'Output': out},
        attrs=op_attr)

    return out


def sequence_expand(x, y):
    """Sequence Expand Layer. This layer will expand the input variable **x**
    according to LoD information of **y**. And the following examples will
    explain how sequence_expand works:

    .. code-block:: text

        * Case 1
            x is a LoDTensor:
                x.lod = [[0,       2, 3],
                         [0, 1,    3, 4]]
                x.data = [a, b, c, d]
                x.dims = [4, 1]

            y is a LoDTensor:
                y.lod = [[0,    2,    4],
                         [0, 3, 6, 7, 8]]

            with condition len(y.lod[-1]) - 1 == x.dims[0]

            then output is a 2-level LoDTensor:
                out.lod = [[0,                2,    4],
                           [0,       3,       6, 7, 8]]
                out.data = [a, a, a, b, b, b, c, d]
                out.dims = [8, 1]

        * Case 2
            x is a Tensor:
                x.data = [a, b, c]
                x.dims = [3, 1]

            y is a LoDTensor:
                y.lod = [[0, 2, 3, 6]]

            with condition len(y.lod[-1]) - 1 == x.dims[0]

            then output is a 1-level LoDTensor:
                out.lod = [[0,    2, 3,      6]]
                out.data = [a, a, b, c, c, c]
                out.dims = [6, 1]

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        y (Variable): The input variable which is a LoDTensor.

    Returns:
        Variable: The expanded variable which is a LoDTensor.

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[10], dtype='float32')
            y = fluid.layers.data(name='y', shape=[10, 20],
                             dtype='float32', lod_level=1)
            out = layers.sequence_expand(x=x, y=y)
    """
    helper = LayerHelper('sequence_expand', input=x, **locals())
    dtype = helper.input_dtype()
    tmp = helper.create_tmp_variable(dtype)
    helper.append_op(
        type='sequence_expand', inputs={'X': x,
                                        'Y': y}, outputs={'Out': tmp})
    return tmp


def lstm_unit(x_t,
              hidden_t_prev,
              cell_t_prev,
              forget_bias=0.0,
              param_attr=None,
              bias_attr=None):
    """Lstm unit layer. The equation of a lstm step is:

        .. math::

            i_t & = \sigma(W_{x_i}x_{t} + W_{h_i}h_{t-1} + W_{c_i}c_{t-1} + b_i)

            f_t & = \sigma(W_{x_f}x_{t} + W_{h_f}h_{t-1} + W_{c_f}c_{t-1} + b_f)

            c_t & = f_tc_{t-1} + i_t tanh (W_{x_c}x_t+W_{h_c}h_{t-1} + b_c)

            o_t & = \sigma(W_{x_o}x_{t} + W_{h_o}h_{t-1} + W_{c_o}c_t + b_o)

            h_t & = o_t tanh(c_t)

    The inputs of lstm unit includes :math:`x_t`, :math:`h_{t-1}` and
    :math:`c_{t-1}`. The implementation separates the linear transformation
    and non-linear transformation apart. Here, we take :math:`i_t` as an
    example. The linear transformation is applied by calling a `fc` layer and
    the equation is:

        .. math::

            L_{i_t} = W_{x_i}x_{t} + W_{h_i}h_{t-1} + W_{c_i}c_{t-1} + b_i

    The non-linear transformation is applied by calling `lstm_unit_op` and the
    equation is:

        .. math::

            i_t = \sigma(L_{i_t})

    This layer has two outputs including :math:`h_t` and :math:`o_t`.

    Args:
        x_t (Variable): The input value of current step.
        hidden_t_prev (Variable): The hidden value of lstm unit.
        cell_t_prev (Variable): The cell value of lstm unit.
        forget_bias (float): The forget bias of lstm unit.
        param_attr (ParamAttr): The attributes of parameter weights, used to set
            initializer, name etc.
        bias_attr (ParamAttr): The attributes of bias weights, if not False,
            bias weights will be created and be set to default value.

    Returns:
        tuple: The hidden value and cell value of lstm unit.

    Raises:
        ValueError: The ranks of **x_t**, **hidden_t_prev** and **cell_t_prev**\
                not be 2 or the 1st dimensions of **x_t**, **hidden_t_prev** \
                and **cell_t_prev** not be the same.

    Examples:

        .. code-block:: python

             x_t = fluid.layers.fc(input=x_t_data, size=10)
             prev_hidden = fluid.layers.fc(input=prev_hidden_data, size=20)
             prev_cell = fluid.layers.fc(input=prev_cell_data, size=30)
             hidden_value, cell_value = fluid.layers.lstm_unit(x_t=x_t,
                                                    hidden_t_prev=prev_hidden,
                                                    cell_t_prev=prev_cell)
    """
    helper = LayerHelper('lstm_unit', **locals())

    if len(x_t.shape) != 2:
        raise ValueError("Rank of x_t must be 2.")

    if len(hidden_t_prev.shape) != 2:
        raise ValueError("Rank of hidden_t_prev must be 2.")

    if len(cell_t_prev.shape) != 2:
        raise ValueError("Rank of cell_t_prev must be 2.")

    if x_t.shape[0] != hidden_t_prev.shape[0] or x_t.shape[
            0] != cell_t_prev.shape[0]:
        raise ValueError("The 1s dimension of x_t, hidden_t_prev and "
                         "cell_t_prev must be the same.")

    if bias_attr is None:
        bias_attr = ParamAttr()

    size = cell_t_prev.shape[1]
    concat_out = concat(input=[x_t, hidden_t_prev], axis=1)
    fc_out = fc(input=concat_out,
                size=4 * size,
                param_attr=param_attr,
                bias_attr=bias_attr)
    dtype = x_t.dtype
    c = helper.create_tmp_variable(dtype)
    h = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='lstm_unit',
        inputs={"X": fc_out,
                "C_prev": cell_t_prev},
        outputs={"C": c,
                 "H": h},
        attrs={"forget_bias": forget_bias})

    return h, c


def reduce_sum(input, dim=None, keep_dim=False):
    """
    Computes the sum of tensor elements over the given dimension. 

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (int|None): The dimension along which the sum is performed. If 
            :attr:`None`, sum all elements of :attr:`input` and return a 
            Tensor variable with a single element, otherwise must be in the 
            range :math:`[-rank(input), rank(input))`. If :math:`dim < 0`, 
            the dimension to reduce is :math:`rank + dim`.
        keep_dim (bool): Whether to reserve the reduced dimension in the 
            output Tensor. The result tensor will have one fewer dimension 
            than the :attr:`input` unless :attr:`keep_dim` is true.

    Returns:
        Variable: The reduced Tensor variable.
    
    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_sum(x)  # [3.5]
            fluid.layers.reduce_sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
            fluid.layers.reduce_sum(x, dim=-1)  # [1.9, 1.6]
            fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]
    """
    helper = LayerHelper('reduce_sum', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='reduce_sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else 0,
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_mean(input, dim=None, keep_dim=False):
    """
    Computes the mean of tensor elements over the given dimension. 

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (int|None): The dimension along which the mean is computed. If 
            :attr:`None`, compute the mean over all elements of :attr:`input` 
            and return a Tensor variable with a single element, otherwise 
            must be in the range :math:`[-rank(input), rank(input))`. If 
            :math:`dim < 0`, the dimension to reduce is :math:`rank + dim`.
        keep_dim (bool): Whether to reserve the reduced dimension in the 
            output Tensor. The result tensor will have one fewer dimension 
            than the :attr:`input` unless :attr:`keep_dim` is true.

    Returns:
        Variable: The reduced Tensor variable.
    
    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_mean(x)  # [0.4375]
            fluid.layers.reduce_mean(x, dim=0)  # [0.15, 0.25, 0.55, 0.8]
            fluid.layers.reduce_mean(x, dim=-1)  # [0.475, 0.4]
            fluid.layers.reduce_mean(x, dim=1, keep_dim=True)  # [[0.475], [0.4]]
    """
    helper = LayerHelper('reduce_mean', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='reduce_mean',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else 0,
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out
