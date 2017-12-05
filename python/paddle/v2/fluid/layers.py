import core
import proto.framework_pb2 as framework_pb2
from framework import OpProtoHolder, Variable, Program, Operator
from initializer import Constant, Normal, Xavier, Initializer
from paddle.v2.fluid.layer_helper import LayerHelper, unique_name
import re
import cStringIO
from param_attr import ParamAttr

__all__ = [
    'fc', 'data', 'cross_entropy', 'conv2d', 'pool2d', 'embedding', 'concat',
    'StaticRNN', 'cast', 'sequence_conv', 'sequence_pool', 'sums', 'cos_sim',
    'batch_norm', 'accuracy', 'split_lod_tensor', 'While'
]


def fc(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       act=None,
       name=None,
       main_program=None,
       startup_program=None):
    """
    Fully Connected Layer.

    Args:
       input: The input tensor to the function
       size: The size of the layer
       num_flatten_dims: Number of columns in input
       param_attr: The parameters/weights to the FC Layer
       param_initializer: Initializer used for the weight/parameter. If None, XavierInitializer() is used
       bias_attr: The bias parameter for the FC layer
       bias_initializer: Initializer used for the bias. If None, then ConstantInitializer() is used
       act: Activation to be applied to the output of FC layer
       name: Name/alias of the function
       main_program: Name of the main program that calls this
       startup_program: Name of the startup program

    This function can take in multiple inputs and performs the Fully Connected
    function (linear transformation) on top of each of them.
    So for input x, the output will be : Wx + b. Where W is the parameter,
    b the bias and x is the input.

    The function also applies an activation (non-linearity) on top of the
    output, if activation is passed in the input.

    All the input variables of this function are passed in as local variables
    to the LayerHelper constructor.

    """
    helper = LayerHelper('fc', **locals())

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
            attrs={'x_num_col_dims': num_flatten_dims,
                   'y_num_col_dims': 1})
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


def embedding(input,
              size,
              is_sparse=False,
              param_attr=None,
              dtype='float32',
              main_program=None,
              startup_program=None):
    """
    Embedding Layer.

    Args:
       param_initializer:
       input: The input to the function
       size: The size of the layer
       is_sparse: A flag that decleares whether the input is sparse
       param_attr: Parameters for this layer
       dtype: The type of data : float32, float_16, int etc
       main_program: Name of the main program that calls this
       startup_program: Name of the startup program

    This function can take in the input (which is a vector of IDs) and
    performs a lookup in the lookup_table using these IDs, to result into
    the embedding of each ID in the input.

    All the input variables of this function are passed in as local variables
    to the LayerHelper constructor.

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
                 dtype='float32',
                 main_program=None,
                 startup_program=None):
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


def data(name,
         shape,
         append_batch_size=True,
         dtype='float32',
         lod_level=0,
         type=core.VarDesc.VarType.LOD_TENSOR,
         main_program=None,
         startup_program=None,
         stop_gradient=True):
    """
    Data Layer.

    Args:
       name: The name/alias of the function
       shape: Tuple declaring the shape.
       append_batch_size: Whether or not to append the data as a batch.
       dtype: The type of data : float32, float_16, int etc
       type: The output type. By default it is LOD_TENSOR.
       lod_level(int): The LoD Level. 0 means the input data is not a sequence.
       main_program: Name of the main program that calls this
       startup_program: Name of the startup program
       stop_gradient: A boolean that mentions whether gradient should flow.

    This function takes in input and based on whether data has
    to be returned back as a minibatch, it creates the global variable using
    the helper functions. The global variables can be accessed by all the
    following operations and layers in the graph.

    All the input variables of this function are passed in as local variables
    to the LayerHelper constructor.

    """
    helper = LayerHelper('data', **locals())
    shape = list(shape)
    for i in xrange(len(shape)):
        if shape[i] is None:
            shape[i] = -1
            append_batch_size = False
        elif shape[i] < 0:
            append_batch_size = False

    if append_batch_size:
        shape = [-1] + shape  # append batch size as -1

    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=type,
        stop_gradient=stop_gradient,
        lod_level=lod_level)


def create_tensor(dtype, name=None, main_program=None, startup_program=None):
    helper = LayerHelper("create_tensor", **locals())
    return helper.create_variable(name=helper.name, dtype=dtype)


def _convert_(name):
    """
    Formatting.

    Args:
       name: The name/alias

    This function takes in a name and converts it to a standard format of
    group1_group2. Where as per the regular expression, group1 can have
    alphabets and numbers and group2 has capital alphabets.

    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _generate_doc_string_(op_proto):
    """
    Generate docstring by OpProto

    Args:
        op_proto (framework_pb2.OpProto): a protobuf message typed OpProto

    Returns:
        str: the document string
    """

    def _type_to_str_(tp):
        return framework_pb2.AttrType.Name(tp)

    if not isinstance(op_proto, framework_pb2.OpProto):
        raise TypeError("OpProto should be `framework_pb2.OpProto`")

    buf = cStringIO.StringIO()
    buf.write(op_proto.comment)
    buf.write('\nArgs:\n')
    for each_input in op_proto.inputs:
        line_begin = '    {0}: '.format(_convert_(each_input.name))
        buf.write(line_begin)
        buf.write(each_input.comment)
        buf.write('\n')
        buf.write(' ' * len(line_begin))
        buf.write('Duplicable: ')
        buf.write(str(each_input.duplicable))
        buf.write('  Optional: ')
        buf.write(str(each_input.dispensable))
        buf.write('\n')

    for each_attr in op_proto.attrs:
        buf.write('    ')
        buf.write(each_attr.name)
        buf.write(' (')
        buf.write(_type_to_str_(each_attr.type))
        buf.write('): ')
        buf.write(each_attr.comment)
        buf.write('\n')

    if len(op_proto.outputs) != 0:
        buf.write('\nReturns:\n')
        buf.write('    ')
        for each_opt in op_proto.outputs:
            if not each_opt.intermediate:
                break
        buf.write(each_opt.comment)

    return buf.getvalue()


def _create_op_func_(op_type):
    """
    Create an Operator for a Function.

    Args:
       op_type: The name of the operator to be created

    This function takes in the operator type (sigmoid, mean , average etc) and
    creates the operator functionality.

    """
    op_proto = OpProtoHolder.instance().get_op_proto(op_type)
    not_intermediate_outputs = \
        filter(lambda output: not output.intermediate, op_proto.outputs)
    intermediate_outputs = \
        filter(lambda output: output.intermediate, op_proto.outputs)

    if len(not_intermediate_outputs) != 1:
        raise ValueError("Only one non intermediate output operator can be",
                         "automatically generated")

    if not_intermediate_outputs[0].duplicable:
        raise ValueError(
            "Only non duplicable op can be automatically generated")

    for output in intermediate_outputs:
        if output.duplicable:
            raise ValueError("The op can be automatically generated only when ",
                             "all intermediate ops are not duplicable")

    o_name = not_intermediate_outputs[0].name
    intermediate_output_names = [output.name for output in intermediate_outputs]

    def infer_and_check_dtype(op_proto, **kwargs):
        """
        This function performs the sanity check for dtype and
        instance type.
        """
        dtype = None
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            for each in val:
                if not isinstance(each, Variable):
                    raise ValueError("input of {0} must be variable".format(
                        op_type))

                if dtype is None:
                    dtype = each.dtype
                elif dtype != each.dtype:
                    raise ValueError(
                        "operator {0} must input same dtype".format(op_type))

        return dtype

    def func(**kwargs):
        helper = LayerHelper(op_type, **kwargs)

        dtype = infer_and_check_dtype(op_proto, **kwargs)

        inputs = dict()
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            inputs[ipt.name] = val

        outputs = dict()
        out = helper.create_tmp_variable(dtype=dtype)
        outputs[o_name] = [out]
        for name in intermediate_output_names:
            outputs[name] = [helper.create_tmp_variable(dtype=dtype)]
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=kwargs)
        return helper.append_activation(out)

    func.__name__ = op_type
    globals()[op_type] = func
    func.__doc__ = _generate_doc_string_(op_proto)
    global __all__
    __all__.append(op_type)


_create_op_func_('mean')
_create_op_func_('mul')
_create_op_func_('elementwise_add')
_create_op_func_('elementwise_div')
_create_op_func_('dropout')
_create_op_func_('reshape')
_create_op_func_('sigmoid')
_create_op_func_('scale')
_create_op_func_('reshape')
_create_op_func_('transpose')
_create_op_func_('sigmoid_cross_entropy_with_logits')


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


def linear_chain_crf(input,
                     label,
                     param_attr=None,
                     main_program=None,
                     startup_program=None):
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


def assign(input, output, main_program=None, startup_program=None):
    helper = LayerHelper('assign', **locals())
    helper.append_op(
        type='scale',
        inputs={'X': [input]},
        outputs={'Out': [output]},
        attrs={'scale': 1.0})
    return output


def split_lod_tensor(input,
                     mask,
                     level=0,
                     main_program=None,
                     startup_program=None):
    helper = LayerHelper('split_lod_tensor', **locals())
    out_true = helper.create_tmp_variable(dtype=input.dtype)
    out_false = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='split_lod_tensor',
        inputs={
            'X': input,
            'Mask': mask,
        },
        outputs={'OutTrue': out_true,
                 'OutFalse': out_false},
        attrs={'level': level})
    return out_true, out_false


def merge_lod_tensor(in_true,
                     in_false,
                     x,
                     mask,
                     level=0,
                     main_program=None,
                     startup_program=None):
    helper = LayerHelper('merge_lod_tensor', **locals())
    out = helper.create_tmp_variable(dtype=in_true.dtype)
    helper.append_op(
        type='merge_lod_tensor',
        inputs={'X': x,
                'Mask': mask,
                'InTrue': in_true,
                'InFalse': in_false},
        outputs={'Out': out},
        attrs={'level': level})
    return out


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


def sequence_conv(input,
                  num_filters,
                  filter_size=3,
                  filter_stride=1,
                  padding=None,
                  bias_attr=None,
                  param_attr=None,
                  act=None,
                  main_program=None,
                  startup_program=None):
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
    filter = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    pre_bias = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='sequence_conv',
        inputs={
            'X': [input],
            'Filter': [filter],
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
           stride=[1, 1],
           padding=None,
           groups=None,
           param_attr=None,
           bias_attr=None,
           act=None,
           name=None,
           main_program=None,
           startup_program=None):
    """
    This function creates the op for a 2-dimensional Convolution.
    This is performed using the parameters of filters(size, dimensionality etc)
    , stride and other configurations for a Convolution operation.
    This funciton can also append an activation on top of the
    conv-2d output, if mentioned in the input parameters.
    """

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

    filter = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='conv2d',
        inputs={
            'Input': input,
            'Filter': filter,
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
    This is applied on top of the input using pool_type mentioned
    in the parameters.
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


def pool2d(input,
           pool_size,
           pool_type,
           pool_stride=[1, 1],
           pool_padding=[0, 0],
           global_pooling=False,
           main_program=None,
           startup_program=None):
    """
    This function adds the operator for pooling in 2 dimensions, using the
    pooling configurations mentioned in input parameters.
    """
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
               data_layout='NCHW',
               main_program=None,
               startup_program=None):
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


def beam_search_decode(ids, scores, main_program=None, startup_program=None):
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


class BlockGuard(object):
    """
    BlockGuard class.

    BlockGuard class is used to create a sub-block in a program by
    using the Python `with` keyword.
    """

    def __init__(self, main_program):
        if not isinstance(main_program, Program):
            raise TypeError("BlockGuard takes a program")
        self.main_program = main_program

    def __enter__(self):
        self.main_program.create_block()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.main_program.rollback()
        if exc_type is not None:
            return False  # re-raise exception
        return True


class StaticRNNGuard(BlockGuard):
    """
    StaticRNNGuard class.

    StaticRNNGuard class is used to create a StaticRNN block in a program.
    """

    def __init__(self, rnn):
        if not isinstance(rnn, StaticRNN):
            raise TypeError("StaticRNNGuard takes a StaticRNN")
        super(StaticRNNGuard, self).__init__(rnn.helper.main_program)
        self.rnn = rnn

    def __enter__(self):
        self.rnn.status = StaticRNN.IN_RNN_BLOCK
        return super(StaticRNNGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.rnn.status = StaticRNN.AFTER_RNN_BLOCK
        self.rnn.complete_rnn_op()
        return super(StaticRNNGuard, self).__exit__(exc_type, exc_val, exc_tb)


class StaticRNNMemoryLink(object):
    """
    StaticRNNMemoryLink class.

    Args:
        init: the initial variable for Memory
        init: Variable
        pre_mem: the memory variable in previous time step
        pre_mem: Variable
        mem: the memory variable in current time step
        mem: Variable

    StaticRNNMemoryLink class is used to create a link between two
    memory cells of a StaticRNN.
    """

    def __init__(self, init, pre_mem, mem=None):
        self.init = init
        self.pre_mem = pre_mem
        self.mem = mem


class StaticRNN(object):
    """
    StaticRNN class.

    StaticRNN class is used to create a StaticRNN. The RNN will have its
    own parameters like inputs, outputs, memories, status and length.
    """
    BEFORE_RNN_BLOCK = 0
    IN_RNN_BLOCK = 1
    AFTER_RNN_BLOCK = 2

    def __init__(self, name=None, main_program=None):
        self.helper = LayerHelper(
            "static_rnn", name=name, main_program=main_program)
        self.memories = {}  # memory map, from pre_mem.name --> MemoryLink
        self.inputs = []  # input variable list in current block
        self.outputs = []  # output variable list in parent block
        self.status = StaticRNN.BEFORE_RNN_BLOCK  # status flag.
        # sequence length, since it is a static RNN, sequence length are fixed.
        self.seq_len = None

    def step(self):
        return StaticRNNGuard(self)

    def _assert_in_rnn_block_(self, method):
        if self.status != StaticRNN.IN_RNN_BLOCK:
            raise ValueError("You must invoke {0} in rnn block".format(method))

    def memory(self,
               init=None,
               shape=None,
               batch_ref=None,
               init_value=0.0,
               init_batch_dim_idx=0,
               ref_batch_dim_idx=1):
        """
        Args:
            init: boot memory, if not set, a shape, batch_ref must be provided
            shape: shape of the boot memory
            batch_ref: batch size reference variable
            init_value: the init value of boot memory
            init_batch_dim_idx: the index of batch size in init's dimension
            ref_batch_dim_idx: the index of batch size in batch_ref's dimension
        """
        self._assert_in_rnn_block_('memory')
        if init is None:
            if shape is None or batch_ref is None:
                raise ValueError(
                    "if init is None, memory at least need shape and batch_ref")
            parent_block = self.parent_block()
            var_name = unique_name("@".join([self.helper.name, "memory_boot"]))
            boot_var = parent_block.create_var(
                name=var_name,
                shape=shape,
                dtype=batch_ref.dtype,
                persistable=False)

            parent_block.append_op(
                type="fill_constant_batch_size_like",
                inputs={'Input': [batch_ref]},
                outputs={'Out': [boot_var]},
                attrs={
                    'value': init_value,
                    'shape': boot_var.shape,
                    'dtype': boot_var.dtype,
                    'input_dim_idx': ref_batch_dim_idx,
                    'output_dim_idx': init_batch_dim_idx
                })

            return self.memory(init=boot_var)
        else:
            pre_mem = self.helper.create_variable(
                name=unique_name("@".join([self.helper.name, "mem"])),
                dtype=init.dtype,
                shape=init.shape)
            self.memories[pre_mem.name] = StaticRNNMemoryLink(
                init=init, pre_mem=pre_mem)
            return pre_mem

    def step_input(self, x):
        self._assert_in_rnn_block_('step_input')
        if not isinstance(x, Variable):
            raise TypeError("step input takes a Variable")
        if self.seq_len is None:
            self.seq_len = x.shape[0]
        elif self.seq_len != x.shape[0]:
            raise ValueError("Static RNN only take fix seq_len input")

        ipt = self.helper.create_variable(
            name=x.name, dtype=x.dtype, shape=list(x.shape[1:]), type=x.type)
        self.inputs.append(ipt)
        return ipt

    def step_output(self, o):
        self._assert_in_rnn_block_('step_output')
        if not isinstance(o, Variable):
            raise TypeError("step output takes a Variable")

        tmp_o = self.helper.create_tmp_variable(dtype=o.dtype)
        self.helper.append_op(
            type='rnn_memory_helper',
            inputs={'X': [o]},
            outputs={'Out': tmp_o},
            attrs={'dtype': o.dtype})

        out_var = self.parent_block().create_var(
            name=tmp_o.name,
            shape=[self.seq_len] + list(tmp_o.shape),
            dtype=tmp_o.dtype)

        self.outputs.append(out_var)

    def output(self, *outputs):
        for each in outputs:
            self.step_output(each)

    def update_memory(self, mem, var):
        if not isinstance(mem, Variable) or not isinstance(var, Variable):
            raise TypeError("update memory should take variables")
        self.memories[mem.name].mem = var

    def parent_block(self):
        prog = self.helper.main_program
        parent_idx = prog.current_block().parent_idx
        assert parent_idx >= 0
        parent_block = prog.block(parent_idx)
        return parent_block

    def __call__(self, *args, **kwargs):
        if self.status != StaticRNN.AFTER_RNN_BLOCK:
            raise ValueError("RNN output can only be retrieved after rnn block")
        if len(self.outputs) == 0:
            raise ValueError("RNN has no output")
        elif len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    def complete_rnn_op(self):
        main_program = self.helper.main_program
        rnn_block = main_program.current_block()
        parent_block = self.parent_block()

        local_inputs = set()

        for op in rnn_block.ops:
            assert isinstance(op, Operator)
            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    local_inputs.add(out_var_name)

        for var in self.inputs:
            local_inputs.add(var.name)
        for m in self.memories:
            local_inputs.add(m)

        params = list()
        for op in rnn_block.ops:
            assert isinstance(op, Operator)
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if in_var_name not in local_inputs:
                        params.append(in_var_name)

        parameters = [parent_block.var(name) for name in params]

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        inlinks = [parent_block.var(i.name) for i in self.inputs]
        outlinks = self.outputs

        boot_memories = []
        pre_memories = []
        memories = []
        for _, mem in self.memories.iteritems():
            boot_memories.append(mem.init)
            pre_memories.append(mem.pre_mem.name)
            mem_var = rnn_block.var(mem.mem.name)
            assert isinstance(mem_var, Variable)
            new_mem = self.helper.create_tmp_variable(dtype=mem_var.dtype)

            rnn_block.append_op(
                type='rnn_memory_helper',
                inputs={'X': [mem_var]},
                outputs={'Out': [new_mem]},
                attrs={'dtype': mem_var.dtype})

            memories.append(new_mem.name)

        parent_block.append_op(
            type='recurrent',
            inputs={
                'inputs': inlinks,
                'initial_states': boot_memories,
                'parameters': parameters
            },
            outputs={'outputs': outlinks,
                     'step_scopes': [step_scope]},
            attrs={
                'ex_states': pre_memories,
                'states': memories,
                'step_block': rnn_block
            })


class WhileGuard(BlockGuard):
    def __init__(self, while_op):
        if not isinstance(while_op, While):
            raise TypeError("WhileGuard takes a while op")
        super(WhileGuard, self).__init__(while_op.helper.main_program)
        self.while_op = while_op

    def __enter__(self):
        self.while_op.status = While.IN_WHILE_BLOCK
        return super(WhileGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.while_op.status = While.AFTER_WHILE_BLOCK
        self.while_op.complete()
        return super(WhileGuard, self).__exit__(exc_type, exc_val, exc_tb)


class While(object):
    BEFORE_WHILE_BLOCK = 0
    IN_WHILE_BLOCK = 1
    AFTER_WHILE_BLOCK = 2

    def __init__(self, cond, name=None, main_program=None):
        self.helper = LayerHelper("while", name=name, main_program=main_program)
        self.status = While.BEFORE_WHILE_BLOCK
        if not isinstance(cond, Variable):
            raise TypeError("condition should be a variable")
        assert isinstance(cond, Variable)
        if cond.dtype != core.DataType.BOOL:
            raise TypeError("condition should be a bool variable")
        if reduce(lambda a, b: a * b, cond.shape, 1) != 1:
            raise TypeError("condition should be a bool scalar")
        self.cond_var = cond

    def block(self):
        return WhileGuard(self)

    def complete(self):
        main_program = self.helper.main_program
        while_block = main_program.current_block()
        parent_block = main_program.block(main_program.current_block()
                                          .parent_idx)

        inner_outputs = {self.cond_var.name}
        x_name_list = set()
        for op in while_block.ops:
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if in_var_name not in inner_outputs:
                        x_name_list.add(in_var_name)

            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    inner_outputs.add(out_var_name)

        out_vars = []
        for inner_out_name in inner_outputs:
            if inner_out_name in parent_block.vars:
                out_vars.append(parent_block.var(inner_out_name))

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        parent_block.append_op(
            type='while',
            inputs={
                'X': [parent_block.var(x_name) for x_name in x_name_list],
                'Condition': [self.cond_var]
            },
            outputs={'Out': out_vars,
                     'StepScopes': [step_scope]},
            attrs={'step_block': while_block})


def lstm(x,
         c_pre_init,
         hidden_dim,
         forget_bias=None,
         main_program=None,
         startup_program=None):
    """
    This function helps create an operator for the LSTM (Long Short Term
    Memory) cell that can be used inside an RNN.
    """
    helper = LayerHelper('lstm_unit', **locals())
    rnn = StaticRNN()
    with rnn.step():
        c_pre = rnn.memory(init=c_pre_init)
        x_t = rnn.step_input(x)

        before_fc = concat(
            input=[x_t, c_pre],
            axis=1,
            main_program=main_program,
            startup_program=startup_program)
        after_fc = fc(input=before_fc,
                      size=hidden_dim * 4,
                      main_program=main_program,
                      startup_program=startup_program)

        dtype = x.dtype
        c = helper.create_tmp_variable(dtype)
        h = helper.create_tmp_variable(dtype)

        helper.append_op(
            type='lstm_unit',
            inputs={"X": after_fc,
                    "C_prev": c_pre},
            outputs={"C": c,
                     "H": h},
            attrs={"forget_bias": forget_bias})

        rnn.update_memory(c_pre, c)
        rnn.output(h)

    return rnn()


def lod_rank_table(x, level=0, main_program=None):
    """
    This function creates an operator for creating a LOD_RANK_TABLE
    using the input x.
    """
    helper = LayerHelper("lod_rank_table", **locals())
    table = helper.create_variable(
        type=core.VarDesc.VarType.LOD_RANK_TABLE,
        name=unique_name("lod_rank_table"))
    helper.append_op(
        type='lod_rank_table',
        inputs={'X': x},
        outputs={'Out': table},
        attrs={'level': level})
    return table


def max_sequence_len(rank_table, main_program=None):
    """
    This function creates an operator to calculate the length of
    max seqence through input rank_table(should be a lod_rank_table)
    """
    helper = LayerHelper("max_seqence_len", **locals())
    res = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="max_sequence_len",
        inputs={"RankTable": rank_table},
        outputs={"Out": res})
    return res


def topk(input, k, main_program=None, startup_program=None):
    helper = LayerHelper('topk', **locals())
    topk_out = helper.create_tmp_variable(dtype=input.data_type)
    topk_indices = helper.create_tmp_variable(dtype='int64')
    helper.append_op(
        type='top_k',
        inputs={'X': [input]},
        outputs={'Out': [topk_out],
                 'Indices': [topk_indices]},
        attrs={'k': k})
    return topk_out, topk_indices


def lod_tensor_to_array(x, table, main_program=None):
    """
    This function creates an operator to convert an LOD_Tensor to
    an array.
    """
    helper = LayerHelper("lod_tensor_to_array", **locals())
    array = helper.create_variable(
        name=unique_name("lod_tensor_to_array"),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype=x.dtype)
    helper.append_op(
        type='lod_tensor_to_array',
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': array})
    return array


def array_to_lod_tensor(x, table, main_program=None):
    """
    This function creates an operator to convert an array to a
    LOD_Tensor.
    """
    helper = LayerHelper("array_to_lod_tensor", **locals())
    tmp = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type="array_to_lod_tensor",
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': tmp})
    return tmp


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


def increment(x, value=1.0, in_place=True, main_program=None):
    """
    This function creates an operator to increment each value in the input
    `x` by an amount: `value` as mentioned in the input parameter. This
    operation is performed in-place by default.
    """
    helper = LayerHelper("increment", **locals())
    if not in_place:
        out = helper.create_tmp_variable(dtype=x.dtype)
    else:
        out = x
    helper.append_op(
        type='increment',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'step': float(value)})
    return out


def array_write(x, i, array=None, main_program=None):
    """
    This function creates an operator to write the data out as a
    LOD_TENSOR_ARRAY.
    """
    helper = LayerHelper('array_write', **locals())
    if array is None:
        array = helper.create_variable(
            name="{0}.out".format(helper.name),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=x.dtype)
    helper.append_op(
        type='write_to_array',
        inputs={'X': [x],
                'I': [i]},
        outputs={'Out': [array]})
    return array


def create_array(dtype, main_program=None):
    helper = LayerHelper("array", **locals())
    return helper.create_variable(
        name="{0}.out".format(helper.name),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype=dtype)


def less_than(x, y, cond=None, main_program=None, **ignored):
    helper = LayerHelper("less_than", **locals())
    if cond is None:
        cond = helper.create_tmp_variable(dtype='bool')
        cond.stop_gradient = True

    helper.append_op(
        type='less_than', inputs={'X': [x],
                                  'Y': [y]}, outputs={'Out': [cond]})
    return cond


def array_read(array, i, main_program=None):
    """
    This function creates an operator to read the data in as a
    LOD_TENSOR_ARRAY.
    """
    helper = LayerHelper('array_read', **locals())
    if not isinstance(
            array,
            Variable) or array.type != core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        raise TypeError("array should be tensor array vairable")
    out = helper.create_tmp_variable(dtype=array.dtype)
    helper.append_op(
        type='read_from_array',
        inputs={'X': [array],
                'I': [i]},
        outputs={'Out': [out]})
    return out


def shrink_memory(x, i, table, main_program=None):
    """
    This function creates an operator to shrink_rnn_memory using the RankTable
    as mentioned in the input parameter.
    """
    helper = LayerHelper('shrink_memory', **locals())
    out = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type='shrink_rnn_memory',
        inputs={'X': [x],
                'I': [i],
                'RankTable': [table]},
        outputs={'Out': [out]},
        attrs={})
    return out


def array_length(array, main_program=None):
    """
    This function creates an operator to find the length of the
    LOD_TENSOR_ARRAY.
    """
    helper = LayerHelper('array_length', **locals())
    tmp = helper.create_tmp_variable(dtype='int64')
    tmp.stop_gradient = True
    helper.append_op(
        type='lod_array_length', inputs={'X': [array]}, outputs={'Out': [tmp]})
    return tmp


def conv2d_transpose(input,
                     num_filters,
                     output_size=None,
                     filter_size=None,
                     padding=None,
                     stride=None,
                     param_attr=None,
                     main_program=None,
                     startup_program=None):
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
        op_attr['strides'] = stride
    elif stride is not None:
        op_attr['strides'] = stride

    if filter_size is None:
        if output_size is None:
            raise ValueError("output_size must be set when filter_size is None")
        if isinstance(output_size, int):
            output_size = [output_size, output_size]

        padding = op_attr.get('paddings', [0, 0])
        stride = op_attr.get('strides', [1, 1])

        h_in = input.shape[2]
        w_in = input.shape[3]
        filter_size_h = output_size[0] - (h_in - 1) * stride[0] + 2 * padding[0]
        filter_size_w = output_size[1] - (w_in - 1) * stride[1] + 2 * padding[1]
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


class ConditionalBlockGuard(BlockGuard):
    def __init__(self, block):
        if not isinstance(block, ConditionalBlock):
            raise TypeError("block should be conditional block")
        super(ConditionalBlockGuard, self).__init__(block.helper.main_program)
        self.block = block

    def __enter__(self):
        return super(ConditionalBlockGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.block.complete()
        return super(ConditionalBlockGuard, self).__exit__(exc_type, exc_val,
                                                           exc_tb)


class ConditionalBlock(object):
    def __init__(self,
                 inputs,
                 name=None,
                 main_program=None,
                 startup_program=None):
        for each_input in inputs:
            if not isinstance(each_input, Variable):
                raise TypeError("Each input should be variable")
        self.inputs = inputs
        self.helper = LayerHelper(
            'conditional_block',
            name=name,
            main_program=main_program,
            startup_program=startup_program)

    def block(self):
        return ConditionalBlockGuard(self)

    def complete(self):
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)

        intermediate = set()
        params = set()

        for each_op in inside_block.ops:
            assert isinstance(each_op, Operator)
            for iname in each_op.input_names:
                for in_var_name in each_op.input(iname):
                    if in_var_name not in intermediate:
                        params.add(in_var_name)

            for oname in each_op.output_names:
                for out_var_name in each_op.output(oname):
                    intermediate.add(out_var_name)
        input_set = set([ipt.name for ipt in self.inputs])

        param_list = [
            parent_block.var(each_name) for each_name in params
            if each_name not in input_set
        ]

        out_list = [
            parent_block.var(var_name) for var_name in parent_block.vars
            if var_name not in intermediate
        ]

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)
        parent_block.append_op(
            type='conditional_block',
            inputs={
                'X': self.inputs,
                'Params': param_list,
            },
            outputs={'Out': out_list,
                     'Scope': [step_scope]},
            attrs={'block': inside_block})


class IfElseBlockGuard(object):
    def __init__(self, is_true, ifelse):
        if not isinstance(ifelse, IfElse):
            raise TypeError("ifelse must be an instance of IfElse class")

        if ifelse.status != IfElse.OUT_IF_ELSE_BLOCKS:
            raise ValueError("You cannot invoke IfElse.block() inside a block")

        self.is_true = is_true
        self.ie = ifelse
        if is_true:
            self.cond_block = ifelse.conditional_true_block
        else:
            self.cond_block = ifelse.conditional_false_block

        if not isinstance(self.cond_block, ConditionalBlock):
            raise TypeError("Unexpected situation")

        self.cond_block = self.cond_block.block()

    def __enter__(self):
        self.ie.status = IfElse.IN_IF_ELSE_TRUE_BLOCKS if self.is_true else IfElse.IN_IF_ELSE_FALSE_BLOCKS
        self.cond_block.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.cond_block.__exit__(exc_type, exc_val, exc_tb):
            # re-raise inside exception
            return False
        if len(self.ie.output_table[1 if self.is_true else 0]) == 0:
            raise ValueError("Must set output inside block")
        self.ie.status = IfElse.OUT_IF_ELSE_BLOCKS


class IfElse(object):
    OUT_IF_ELSE_BLOCKS = 0
    IN_IF_ELSE_TRUE_BLOCKS = 1
    IN_IF_ELSE_FALSE_BLOCKS = 2

    def __init__(self, cond, name=None, main_program=None,
                 startup_program=None):
        if not isinstance(cond, Variable):
            raise TypeError("cond must be a Variable")
        self.helper = LayerHelper(
            'ifelse',
            name=name,
            main_program=main_program,
            startup_program=startup_program)
        self.cond = cond
        self.input_table = {}
        self.status = IfElse.OUT_IF_ELSE_BLOCKS
        self.conditional_true_block = ConditionalBlock(inputs=[self.cond])
        self.conditional_false_block = ConditionalBlock(inputs=[self.cond])
        self.output_table = ([], [])  # (true_outs, false_outs)

    def input(self, x):
        if self.status == IfElse.OUT_IF_ELSE_BLOCKS:
            raise ValueError("input must in true/false blocks")
        if id(x) not in self.input_table:
            parent_block = self.parent_block()
            out_true = parent_block.create_var(
                name=unique_name('ifelse_input' + self.helper.name),
                dtype=x.dtype)

            out_false = parent_block.create_var(
                name=unique_name('ifelse_input' + self.helper.name),
                dtype=x.dtype)
            parent_block.append_op(
                type='split_lod_tensor',
                inputs={
                    'X': x,
                    'Mask': self.cond,
                },
                outputs={'OutTrue': out_true,
                         'OutFalse': out_false},
                attrs={'level': 0})
            self.input_table[id(x)] = (out_true, out_false)
        else:
            out_true, out_false = self.input_table[id(x)]

        if self.status == IfElse.IN_IF_ELSE_TRUE_BLOCKS:
            return out_true
        else:
            return out_false

    def parent_block(self):
        current_block = self.helper.main_program.current_block()
        return self.helper.main_program.block(current_block.parent_idx)

    def true_block(self):
        return IfElseBlockGuard(True, self)

    def false_block(self):
        return IfElseBlockGuard(False, self)

    def output(self, *outs):
        if self.status == self.OUT_IF_ELSE_BLOCKS:
            raise ValueError("output can only be invoked in the sub-block")

        out_table = self.output_table[1 if self.status ==
                                      self.IN_IF_ELSE_TRUE_BLOCKS else 0]
        parent_block = self.parent_block()
        for each_out in outs:
            if not isinstance(each_out, Variable):
                raise TypeError("Each output should be a variable")
            # create outside tensor
            outside_out = parent_block.create_var(
                name=unique_name("_".join([self.helper.name, 'output'])),
                dtype=each_out.dtype)
            out_table.append(outside_out)

            # assign local var to outside
            assign(
                input=each_out,
                output=outside_out,
                main_program=self.helper.main_program,
                startup_program=self.helper.startup_program)

    def __call__(self):
        if self.status != self.OUT_IF_ELSE_BLOCKS:
            raise ValueError("IfElse::__call__ must be out of sub-block")
        false_len, true_len = map(len, self.output_table)
        if false_len == 0 and true_len == 0:
            raise ValueError("Must invoke true_block/false_block before "
                             "__call__")
        elif false_len != true_len and false_len != 0 and true_len != 0:
            raise ValueError("The output side must be same")
        elif false_len == 0 or true_len == 0:
            return self.output_table[0 if false_len != 0 else 1]

        # else none of false_len/true_len is zero
        # merge together
        rlist = []
        for false_var, true_var in zip(*self.output_table):
            rlist.append(
                merge_lod_tensor(
                    in_true=true_var,
                    in_false=false_var,
                    mask=self.cond,
                    x=self.cond,
                    level=0,
                    main_program=self.helper.main_program,
                    startup_program=self.helper.startup_program))
        return rlist
