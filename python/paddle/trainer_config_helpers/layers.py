# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import functools
import collections
import inspect

from paddle.trainer.config_parser import *
from .activations import LinearActivation, SigmoidActivation, TanhActivation, \
    ReluActivation, IdentityActivation, SoftmaxActivation, BaseActivation
from .evaluators import *
from .poolings import MaxPooling, AvgPooling, BasePoolingType
from .attrs import *
from .default_decorators import *

try:
    import cPickle as pickle
except ImportError:
    import pickle
import copy

__all__ = [
    "full_matrix_projection",
    "AggregateLevel",
    "ExpandLevel",
    "identity_projection",
    "dotmul_projection",
    "dotmul_operator",
    "repeat_layer",
    "seq_reshape_layer",
    "table_projection",
    "mixed_layer",
    "data_layer",
    "embedding_layer",
    "fc_layer",
    "grumemory",
    "pooling_layer",
    "lstmemory",
    "last_seq",
    "first_seq",
    "cos_sim",
    "hsigmoid",
    "conv_projection",
    "mse_cost",
    "regression_cost",
    'classification_cost',
    "LayerOutput",
    'img_conv_layer',
    'img_pool_layer',
    'batch_norm_layer',
    'img_cmrnorm_layer',
    'addto_layer',
    'concat_layer',
    'seq_concat_layer',
    'lstm_step_layer',
    'recurrent_group',
    'memory',
    'StaticInput',
    'expand_layer',
    'scaling_layer',
    'scaling_projection',
    'power_layer',
    'interpolation_layer',
    'bilinear_interp_layer',
    'trans_layer',
    'rotate_layer',
    'sum_to_one_norm_layer',
    'get_output_layer',
    'LayerType',
    'context_projection',
    'beam_search',
    'maxid_layer',
    'GeneratedInput',
    'SubsequenceInput',
    'gru_step_layer',
    'gru_step_naive_layer',
    'recurrent_layer',
    'BaseGeneratedInput',
    'conv_operator',
    'conv_shift_layer',
    'tensor_layer',
    'selective_fc_layer',
    'sampling_id_layer',
    'slope_intercept_layer',
    'trans_full_matrix_projection',
    'linear_comb_layer',
    'convex_comb_layer',
    'ctc_layer',
    'warp_ctc_layer',
    'crf_layer',
    'crf_decoding_layer',
    'nce_layer',
    'cross_entropy_with_selfnorm',
    'cross_entropy',
    'multi_binary_label_cross_entropy',
    'sum_cost',
    'rank_cost',
    'lambda_cost',
    'huber_cost',
    'block_expand_layer',
    'maxout_layer',
    'out_prod_layer',
    'print_layer',
    'priorbox_layer',
    'cross_channel_norm_layer',
    'spp_layer',
    'pad_layer',
    'eos_layer',
    'layer_support',
]


class LayerType(object):
    """
    Layer type enumerations.
    """

    DATA = "data"
    MIXED_LAYER = "mixed"
    LSTMEMORY = "lstmemory"
    GRUMEMORY = "gated_recurrent"
    SEQUENCE_LAST_INSTANCE = "seqlastins"
    SEQUENCE_FIRST_INSTANCE = "seqfirstins"
    SEQUENCE_RESHAPE = "seqreshape"
    POOLING_MAX = "max"
    POOLING_AVG = 'average'
    FC_LAYER = "fc"
    COST = 'cost'
    COSINE_SIM_VEC = 'cos_vm'
    COSINE_SIM = 'cos'
    HSIGMOID = 'hsigmoid'
    CONV_LAYER = "conv"
    CONVTRANS_LAYER = "convt"
    EXCONV_LAYER = "exconv"
    EXCONVTRANS_LAYER = "exconvt"
    CUDNNCONV_LAYER = "cudnn_conv"
    POOL_LAYER = "pool"
    BATCH_NORM_LAYER = 'batch_norm'
    NORM_LAYER = 'norm'
    SUM_TO_ONE_NORM_LAYER = 'sum_to_one_norm'
    ADDTO_LAYER = 'addto'

    CONCAT_LAYER = 'concat'
    CONCAT_PROJ_LAYER = 'concat2'
    SEQUENCE_CONCAT_LAYER = 'seqconcat'

    LSTM_STEP_LAYER = 'lstm_step'
    GRU_STEP_LAYER = 'gru_step'
    GET_OUTPUT_LAYER = 'get_output'

    EXPAND_LAYER = 'expand'
    INTERPOLATION_LAYER = 'interpolation'
    BILINEAR_INTERP_LAYER = 'bilinear_interp'
    POWER_LAYER = 'power'
    SCALING_LAYER = 'scaling'
    TRANS_LAYER = 'trans'
    ROTATE_LAYER = 'rotate'
    OUT_PROD_LAYER = 'out_prod'
    FEATURE_MAP_EXPAND_LAYER = 'featmap_expand'

    MEMORY = 'memory'
    MAXID_LAYER = 'maxid'
    EOSID_LAYER = 'eos_id'
    RECURRENT_LAYER = 'recurrent'

    CONV_SHIFT_LAYER = "conv_shift"
    TENSOR_LAYER = "tensor"
    SEL_FC_LAYER = "selective_fc"
    SAMPLING_ID_LAYER = "sampling_id"
    SLOPE_INTERCEPT_LAYER = "slope_intercept"
    LINEAR_COMBINATION_LAYER = "convex_comb"
    BLOCK_EXPAND = "blockexpand"
    MAXOUT = "maxout"
    SPP_LAYER = "spp"
    PAD_LAYER = "pad"

    PRINT_LAYER = "print"
    PRIORBOX_LAYER = "priorbox"

    CTC_LAYER = "ctc"
    WARP_CTC_LAYER = "warp_ctc"
    CRF_LAYER = "crf"
    CRF_DECODING_LAYER = "crf_decoding"
    NCE_LAYER = 'nce'

    RANK_COST = "rank-cost"
    LAMBDA_COST = "lambda_cost"
    HUBER = "huber"
    CROSS_ENTROPY = "multi-class-cross-entropy"
    CROSS_ENTROPY_WITH_SELFNORM = "multi_class_cross_entropy_with_selfnorm"
    SOFT_BIN_CLASS_CROSS_ENTROPY = "soft_binary_class_cross_entropy"
    MULTI_BIN_LABEL_CROSS_ENTROPY = "multi_binary_label_cross_entropy"
    SUM_COST = "sum_cost"

    @staticmethod
    def is_layer_type(type_name):
        """
        If type_name is a layer type.

        :param type_name: layer type name. Because layer type enumerations are
                          strings.
        :type type_name: basestring
        :return: True if is a layer_type
        :rtype: bool
        """
        for key in dir(LayerType):
            if key.isupper():
                att = getattr(LayerType, key)
                if isinstance(att, basestring) and type_name == att:
                    return True
        return False


class AggregateLevel(object):
    EACH_TIMESTEP = 'non-seq'
    EACH_SEQUENCE = 'seq'


class LayerOutput(object):
    """
    LayerOutput is output for layer function. It is used internally by several
    reasons.

    - Check layer connection make sense.

        - FC(Softmax) => Cost(MSE Error) is not good for example.

    - Tracking layer connection.

    - Pass to layer methods as input.

    :param name: Layer output name.
    :type name: basestring
    :param layer_type: Current Layer Type. One of LayerType enumeration.
    :type layer_type: basestring
    :param activation: Layer Activation.
    :type activation: BaseActivation.
    :param parents: Layer's parents.
    :type parents: list|tuple|collections.Sequence
    """

    def __init__(self,
                 name,
                 layer_type,
                 parents=None,
                 activation=None,
                 num_filters=None,
                 img_norm_type=None,
                 size=None,
                 outputs=None,
                 reverse=None):
        assert isinstance(name, basestring)
        assert isinstance(layer_type, basestring)
        assert size is not None
        assert LayerType.is_layer_type(layer_type)
        self.name = name
        self.layer_type = layer_type
        if parents is not None and type(parents) != list:
            parents = [parents]
        self.parents = [] if parents is None else parents
        self.activation = activation
        self.num_filters = num_filters
        self.img_norm_type = img_norm_type
        self.size = size
        if outputs is None:
            outputs = ['default']
        self.outputs = outputs
        self.reverse = reverse

    def __repr__(self):
        """
        Disable __repr__ for debug reason. Will be implemented when release
        """
        assert False, "this method should not be invoked"

    def __str__(self):
        """
        Disable __str__ for debug reason. Will be implemented when release
        """
        assert False, "this method should not be invoked"

    def set_input(self, input):
        """
        Set the input for a memory layer. Can only be used for memory layer
        """
        assert isinstance(input, LayerOutput)
        assert self.layer_type == LayerType.MEMORY
        SetMemoryInput(self.name, input.name)


ERROR_CLIPPING = 'error_clipping_threshold'
DROPOUT = 'drop_rate'
DEVICE = 'device'


def layer_support(*attrs):
    attrs_list = list(attrs)
    attrs_list.append(DEVICE)

    def decorator(method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            for attr in attrs_list:
                for each in args:
                    if isinstance(each, ExtraLayerAttribute):
                        setattr(each, '_'.join(['can', attr]), True)
                for key in kwargs:
                    val = kwargs[key]
                    if isinstance(val, ExtraLayerAttribute):
                        setattr(val, '_'.join(['can', attr]), True)
            for each in args:
                if isinstance(each, ExtraLayerAttribute):
                    each.check(method.__name__)
            for key in kwargs:
                val = kwargs[key]
                if isinstance(val, ExtraLayerAttribute):
                    val.check(method.__name__)
            return method(*args, **kwargs)

        if hasattr(method, 'argspec'):
            wrapper.argspec = method.argspec
        else:
            wrapper.argspec = inspect.getargspec(method)

        return wrapper

    return decorator


@wrap_param_attr_default()
def full_matrix_projection(input, size=0, param_attr=None):
    """
    Full Matrix Projection. It performs full matrix multiplication.

    ..  math::
        out.row[i] += in.row[i] * weight

    There are two styles of usage.

    1. When used in mixed_layer like this, you can only set the input:

    .. code-block:: python

       with mixed_layer(size=100) as m:
           m += full_matrix_projection(input=layer)

    2. When used as an independant object like this, you must set the size:

    .. code-block:: python

       proj = full_matrix_projection(input=layer,
                                     size=100,
                                     param_attr=ParamAttr(name='_proj'))

    :param input: input layer
    :type input: LayerOutput
    :param size: The parameter size. Means the width of parameter.
    :type size: int
    :param param_attr: Parameter config, None if use default.
    :type param_attr: ParameterAttribute
    :return: A FullMatrixProjection Object.
    :rtype: FullMatrixProjection
    """
    proj = FullMatrixProjection(
        input_layer_name=input.name, size=size, **param_attr.attr)
    proj.origin = input
    return proj


@wrap_param_attr_default()
def trans_full_matrix_projection(input, size=0, param_attr=None):
    """
    Different from full_matrix_projection, this projection performs matrix
    multiplication, using transpose of weight.

    ..  math::
        out.row[i] += in.row[i] * w^\mathrm{T}

    :math:`w^\mathrm{T}` means transpose of weight.
    The simply usage is:

    .. code-block:: python

       proj = trans_full_matrix_projection(input=layer,
                                           size=100,
                                           param_attr=ParamAttr(
                                                name='_proj',
                                                initial_mean=0.0,
                                                initial_std=0.01))

    :param input: input layer
    :type input: LayerOutput
    :param size: The parameter size. Means the width of parameter.
    :type size: int
    :param param_attr: Parameter config, None if use default.
    :type param_attr: ParameterAttribute
    :return: A TransposedFullMatrixProjection Object.
    :rtype: TransposedFullMatrixProjection
    """
    proj = TransposedFullMatrixProjection(
        input_layer_name=input.name, size=size, **param_attr.attr)
    proj.origin = input
    return proj


@wrap_param_attr_default()
def table_projection(input, size=0, param_attr=None):
    """
    Table Projection. It selects rows from parameter where row\_id
    is in input\_ids.

    .. math::
       out.row[i] += table.row[ids[i]]

    where :math:`out` is output, :math:`table` is parameter, :math:`ids` is input\_ids,
    and :math:`i` is row\_id.

    There are two styles of usage.

    1. When used in mixed_layer like this, you can only set the input:

    .. code-block:: python

       with mixed_layer(size=100) as m:
           m += table_projection(input=layer)

    2. When used as an independant object like this, you must set the size:

    .. code-block:: python

       proj = table_projection(input=layer,
                               size=100,
                               param_attr=ParamAttr(name='_proj'))


    :param input: Input layer, which must contains id fields.
    :type input: LayerOutput
    :param size: The parameter size. Means the width of parameter.
    :type size: int
    :param param_attr: Parameter config, None if use default.
    :type param_attr: ParameterAttribute
    :return: A TableProjection Object.
    :rtype: TableProjection
    """
    proj = TableProjection(
        input_layer_name=input.name, size=size, **param_attr.attr)
    proj.origin = input
    return proj


def identity_projection(input, offset=None):
    """
    1. IdentityProjection if offset=None. It performs:

    .. math::
       out.row[i] += in.row[i]

    The example usage is:

    .. code-block:: python

       proj = identity_projection(input=layer)


    2. IdentityOffsetProjection if offset!=None. It likes IdentityProjection,
    but layer size may be smaller than input size.
    It select dimesions [offset, offset+layer_size) from input:

    .. math::
       out.row[i] += in.row[i + \\textrm{offset}]

    The example usage is:

    .. code-block:: python

       proj = identity_projection(input=layer,
                                  offset=10)

    Note that both of two projections should not have any parameter.

    :param input: Input Layer.
    :type input: LayerOutput
    :param offset: Offset, None if use default.
    :type offset: int
    :return: A IdentityProjection or IdentityOffsetProjection object
    :rtype: IdentityProjection or IdentityOffsetProjection
    """
    if offset is None:
        proj = IdentityProjection(input_layer_name=input.name)
        proj.origin = input
    else:
        proj = IdentityOffsetProjection(
            input_layer_name=input.name, offset=offset)
        proj.origin = input
    return proj


@wrap_param_attr_default()
def scaling_projection(input, param_attr=None):
    """
    scaling_projection multiplies the input with a scalar parameter and add to
    the output.

    .. math::
       out += w * in

    The example usage is:

    .. code-block:: python

       proj = scaling_projection(input=layer)

    :param input: Input Layer.
    :type input: LayerOutput
    :param param_attr: Parameter config, None if use default.
    :type param_attr: ParameterAttribute
    :return: A ScalingProjection object
    :rtype: ScalingProjection
    """
    proj = ScalingProjection(input_layer_name=input.name, **param_attr.attr)
    proj.origin = input
    return proj


@wrap_param_attr_default()
def dotmul_projection(input, param_attr=None):
    """
    DotMulProjection with a layer as input.
    It performs element-wise multiplication with weight.

    ..  math::
        out.row[i] += in.row[i] .* weight

    where :math:`.*` means element-wise multiplication.

    The example usage is:

    .. code-block:: python

       proj = dotmul_projection(input=layer)

    :param input: Input layer.
    :type input: LayerOutput
    :param param_attr: Parameter config, None if use default.
    :type param_attr: ParameterAttribute
    :return: A DotMulProjection Object.
    :rtype: DotMulProjection
    """
    proj = DotMulProjection(
        input_layer_name=input.name, size=input.size, **param_attr.attr)
    proj.origin = input
    return proj


def dotmul_operator(a=None, b=None, scale=1, **kwargs):
    """
    DotMulOperator takes two inputs and performs element-wise multiplication:

    .. math::
       out.row[i] += scale * (a.row[i] .* b.row[i])

    where :math:`.*` means element-wise multiplication, and
    scale is a config scalar, its default value is one.

    The example usage is:

    .. code-block:: python

       op = dotmul_operator(a=layer1, b=layer2, scale=0.5)

    :param a: Input layer1
    :type a: LayerOutput
    :param b: Input layer2
    :type b: LayerOutput
    :param scale: config scalar, default value is one.
    :type scale: float
    :return: A DotMulOperator Object.
    :rtype: DotMulOperator
    """
    if 'x' in kwargs or 'y' in kwargs:
        logger.warning('x and y arguments for dotmul_operator is deprecated. '
                       'Please use a and b as parameter.')
    a = kwargs.get('x', a)  # For Backward capacity.
    b = kwargs.get('y', b)
    assert isinstance(a, LayerOutput)
    assert isinstance(b, LayerOutput)
    if a.size is not None and b.size is not None:
        assert a.size == b.size

    op = DotMulOperator(input_layer_names=[a.name, b.name], scale=scale)
    op.origin = [a, b]
    return op


@wrap_bias_attr_default(['padding_attr'])
def context_projection(input,
                       context_len,
                       context_start=None,
                       padding_attr=False):
    """
    Context Projection.

    It just simply reorganizes input sequence, combines "context_len" sequence
    to one context from context_start. "context_start" will be set to
    -(context_len - 1) / 2 by default. If context position out of sequence
    length, padding will be filled as zero if padding_attr = False, otherwise
    it is trainable.

    For example, origin sequence is [A B C D E F G], context len is 3, then
    after context projection and not set padding_attr, sequence will
    be [ 0AB ABC BCD CDE DEF EFG FG0 ].

    :param input: Input Sequence.
    :type input: LayerOutput
    :param context_len: context length.
    :type context_len: int
    :param context_start: context start position. Default is
                          -(context_len - 1)/2
    :type context_start: int
    :param padding_attr: Padding Parameter Attribute. If false, it means padding
                         always be zero. Otherwise Padding is learnable, and
                         parameter attribute is set by this parameter.
    :type padding_attr: bool|ParameterAttribute
    :return: Projection
    :rtype: Projection
    """
    context_start = -(
        context_len - 1) / 2 if context_start is None else context_start

    extra_dict = dict()
    trainable = isinstance(padding_attr, ParameterAttribute)
    if trainable:
        extra_dict = padding_attr.attr

    proj = ContextProjection(
        input_layer_name=input.name,
        context_length=context_len,
        context_start=context_start,
        trainable_padding=trainable,
        **extra_dict)
    proj.origin = input
    return proj


class MixedLayerType(LayerOutput):
    """
    The internal object for trainer_helpers.
    """

    class AddToSealedMixedLayerException(Exception):
        def __init__(self):
            Exception.__init__(self)

    def __init__(self, name, size, act, bias_attr, layer_attr, parents=None):
        """
        Ctor.
        :param name: layer name.
        :type name: basestring
        :param size: layer size.
        :type size: int
        :param act: activation type.
        :type act: BaseActivation
        :param bias_attr: The Bias Attribute. If no bias, then pass False or
                          something not type of ParameterAttribute. None will
                          get a default Bias.
        :type bias_attr: ParameterAttribute or None means has bias. Any other
                         type means no bias.
        :param layer_attr: Extra Layer Attribute.
        :type layer_attr: ExtraLayerAttribute or None
        """
        LayerOutput.__init__(
            self,
            name,
            LayerType.MIXED_LAYER,
            parents,
            size=size,
            activation=act)
        self.bias_attr = bias_attr
        self.layer_attr = layer_attr
        self.inputs = []
        self.finalized = False

    def __iadd__(self, other):
        """
        + += operator
        :param other: Other projection.
        :type other: Projection
        :return: self.
        :rtype: MixedLayerType
        """
        if not self.finalized:
            assert isinstance(other, Projection) or isinstance(other, Operator)
            self.inputs.append(other)
            if isinstance(other, Projection):
                self.parents.append(other.origin)
            else:
                self.parents.extend(other.origin)
            return self
        else:
            raise MixedLayerType.AddToSealedMixedLayerException()

    def __enter__(self):
        assert len(self.inputs) == 0
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_value is not None:
            raise exc_value
        assert len(self.inputs) != 0
        ml = MixedLayer(
            name=self.name,
            size=self.size,
            active_type=self.activation.name,
            bias=ParamAttr.to_bias(self.bias_attr),
            inputs=self.inputs,
            **ExtraLayerAttribute.to_kwargs(self.layer_attr))
        # update the size which might be computed inside MixedLayer
        # according to the operator's output size
        self.size = ml.config.size
        self.finalized = True


@wrap_name_default("mixed")
@wrap_act_default(act=LinearActivation())
@wrap_bias_attr_default(has_bias=False)
@layer_support(ERROR_CLIPPING, DROPOUT)
def mixed_layer(size=0,
                input=None,
                name=None,
                act=None,
                bias_attr=False,
                layer_attr=None):
    """
    Mixed Layer. A mixed layer will add all inputs together, then activate.
    Each inputs is a projection or operator.

    There are two styles of usages.

    1. When not set inputs parameter, use mixed_layer like this:

    .. code-block:: python

       with mixed_layer(size=256) as m:
           m += full_matrix_projection(input=layer1)
           m += identity_projection(input=layer2)

    2. You can also set all inputs when invoke mixed_layer as follows:

    .. code-block:: python

       m = mixed_layer(size=256,
                       input=[full_matrix_projection(input=layer1),
                              full_matrix_projection(input=layer2)])

    :param name: mixed layer name. Can be referenced by other layer.
    :type name: basestring
    :param size: layer size.
    :type size: int
    :param input: inputs layer. It is an optional parameter. If set,
                  then this function will just return layer's name.
    :param act: Activation Type.
    :type act: BaseActivation
    :param bias_attr: The Bias Attribute. If no bias, then pass False or
                      something not type of ParameterAttribute. None will get a
                      default Bias.
    :type bias_attr: ParameterAttribute or None or bool
    :param layer_attr: The extra layer config. Default is None.
    :type layer_attr: ExtraLayerAttribute
    :return: MixedLayerType object can add inputs or layer name.
    :rtype: MixedLayerType
    """

    if input is None:
        return MixedLayerType(name, size, act, bias_attr, layer_attr)
    else:
        with mixed_layer(
                name=name,
                size=size,
                act=act,
                bias_attr=bias_attr,
                layer_attr=layer_attr) as m:
            if isinstance(input, collections.Sequence):
                for each in input:
                    m += each
            else:
                m += input
        return m


@layer_support()
def data_layer(name, size, height=None, width=None, layer_attr=None):
    """
    Define DataLayer For NeuralNetwork.

    The example usage is:

    ..  code-block:: python

        data = data_layer(name="input", size=1000)

    :param name: Name of this data layer.
    :type name: basestring
    :param size: Size of this data layer.
    :type size: int
    :param height: Height of this data layer, used for image
    :type height: int|None
    :param width: Width of this data layer, used for image
    :type width: int|None
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    Layer(
        type=LayerType.DATA,
        name=name,
        size=size,
        height=height,
        width=width,
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(name, LayerType.DATA, size=size)


@wrap_name_default("embedding")
@wrap_param_attr_default()
@layer_support(ERROR_CLIPPING)
def embedding_layer(input, size, name=None, param_attr=None, layer_attr=None):
    """
    Define a embedding Layer.

    :param name: Name of this embedding layer.
    :type name: basestring
    :param input: The input layer for this embedding. NOTE: must be Index Data.
    :type input: LayerOutput
    :param size: The embedding dimension.
    :type size: int
    :param param_attr: The embedding parameter attribute. See ParameterAttribute
                      for details.
    :type param_attr: ParameterAttribute|None
    :param layer_attr: Extra layer Config. Default is None.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    with mixed_layer(
            name=name,
            size=size,
            act=LinearActivation(),
            bias_attr=False,
            layer_attr=layer_attr) as mix:
        mix += table_projection(input=input, size=size, param_attr=param_attr)
    return mix


@wrap_name_default()
@wrap_param_attr_default()
@wrap_bias_attr_default()
@wrap_act_default()
@layer_support(ERROR_CLIPPING, DROPOUT)
def fc_layer(input,
             size,
             act=None,
             name=None,
             param_attr=None,
             bias_attr=None,
             layer_attr=None):
    """
    Helper for declare fully connected layer.

    The example usage is:

    .. code-block:: python

       fc = fc_layer(input=layer,
                     size=1024,
                     act=LinearActivation(),
                     bias_attr=False)

    which is equal to:

    .. code-block:: python

       with mixed_layer(size=1024) as fc:
           fc += full_matrix_projection(input=layer)

    :param name: The Layer Name.
    :type name: basestring
    :param input: The input layer. Could be a list/tuple of input layer.
    :type input: LayerOutput|list|tuple
    :param size: The layer dimension.
    :type size: int
    :param act: Activation Type. Default is tanh.
    :type act: BaseActivation
    :param param_attr: The Parameter Attribute|list.
    :type param_attr: ParameterAttribute
    :param bias_attr: The Bias Attribute. If no bias, then pass False or
                      something not type of ParameterAttribute. None will get a
                      default Bias.
    :type bias_attr: ParameterAttribute|None|Any
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if isinstance(input, LayerOutput):
        input = [input]
        assert not isinstance(param_attr, collections.Sequence)
        param_attr = [param_attr]
    else:
        if isinstance(param_attr, collections.Sequence):
            assert len(input) == len(param_attr)
        else:
            param_attr = [copy.deepcopy(param_attr) for _ in range(len(input))]

    assert isinstance(input, collections.Sequence)

    Layer(
        inputs=[
            Input(ipt.name, **attr.attr) for ipt, attr in zip(input, param_attr)
        ],
        name=name,
        type=LayerType.FC_LAYER,
        size=size,
        bias=ParamAttr.to_bias(bias_attr),
        active_type=act.name,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.FC_LAYER, input, activation=act, size=size)


@wrap_name_default("print")
def print_layer(input, name=None):
    """
    Print the output value of input layers. This layer is useful for debugging.

    :param name: The Layer Name.
    :type name: basestring
    :param input: The input layer. Could be a list/tuple of input layer.
    :type input: LayerOutput|list|tuple
    :return: LayerOutput
    """
    if isinstance(input, LayerOutput):
        input = [input]
    assert isinstance(input, collections.Sequence)  # list or tuple
    for each in input:
        assert isinstance(each, LayerOutput)

    Layer(
        name=name,
        type=LayerType.PRINT_LAYER,
        inputs=[l.name for l in input], )
    # this layer don't return anything, can not be input of other layer.


@wrap_name_default("priorbox")
def priorbox_layer(input,
                   image,
                   aspect_ratio,
                   variance,
                   min_size,
                   max_size=[],
                   name=None):
    """
    Compute the priorbox and set the variance. This layer is necessary for ssd.

    :param name: The Layer Name.
    :type name: basestring
    :param input: The input layer.
    :type input: LayerOutput
    :param image: The network input image.
    :type image: LayerOutput
    :param aspect_ratio: The aspect ratio.
    :type aspect_ratio: list
    :param variance: The bounding box variance.
    :type min_size: The min size of the priorbox width/height.
    :param min_size: list
    :type max_size: The max size of the priorbox width/height. Could be NULL.
    :param max_size: list
    :return: LayerOutput
    """
    # plus one for ratio 1.
    num_filters = (len(aspect_ratio) * 2 + 1 + len(max_size)) * 4
    size = (input.size / input.num_filters) * num_filters * 2
    Layer(
        name=name,
        type=LayerType.PRIORBOX_LAYER,
        inputs=[input.name, image.name],
        size=size,
        min_size=min_size,
        max_size=max_size,
        aspect_ratio=aspect_ratio,
        variance=variance)
    return LayerOutput(
        name,
        LayerType.PRIORBOX_LAYER,
        parents=[input, image],
        num_filters=num_filters,
        size=size)


@wrap_name_default("cross_channel_norm")
def cross_channel_norm_layer(input, name=None, param_attr=None):
    """
    Normalize a layer's output. This layer is necessary for ssd.
    This layer applys normalize across the channels of each sample to
    a conv layer's output and scale the output by a group of trainable
    factors which dimensions equal to the channel's number.

    :param name: The Layer Name.
    :type name: basestring
    :param input: The input layer.
    :type input: LayerOutput
    :param param_attr: The Parameter Attribute|list.
    :type param_attr: ParameterAttribute
    :return: LayerOutput
    """
    assert input.num_filters is not None
    Layer(
        name=name,
        type=LayerType.NORM_LAYER,
        inputs=[
            Input(
                input.name,
                norm=Norm(
                    norm_type="cross-channel-norm",
                    channels=input.num_filters,
                    size=input.size,
                    scale=0,
                    pow=0,
                    blocked=0),
                **param_attr.attr)
        ])
    return LayerOutput(
        name,
        LayerType.NORM_LAYER,
        parents=input,
        num_filters=input.num_filters,
        size=input.size)


@wrap_name_default("seq_pooling")
@wrap_bias_attr_default(has_bias=False)
@wrap_param_default(['pooling_type'], default_factory=lambda _: MaxPooling())
@layer_support()
def pooling_layer(input,
                  pooling_type=None,
                  name=None,
                  bias_attr=None,
                  agg_level=AggregateLevel.EACH_TIMESTEP,
                  layer_attr=None):
    """
    Pooling layer for sequence inputs, not used for Image.

    The example usage is:

    .. code-block:: python

       seq_pool = pooling_layer(input=layer,
                                pooling_type=AvgPooling(),
                                agg_level=AggregateLevel.EACH_SEQUENCE)

    :param agg_level: AggregateLevel.EACH_TIMESTEP or
                      AggregateLevel.EACH_SEQUENCE
    :type agg_level: AggregateLevel
    :param name: layer name.
    :type name: basestring
    :param input: input layer name.
    :type input: LayerOutput
    :param pooling_type: Type of pooling, MaxPooling(default), AvgPooling,
                         SumPooling, SquareRootNPooling.
    :type pooling_type: BasePoolingType|None
    :param bias_attr: Bias parameter attribute. False if no bias.
    :type bias_attr: ParameterAttribute|None|False
    :param layer_attr: The Extra Attributes for layer, such as dropout.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    extra_dict = dict()
    # noinspection PyUnresolvedReferences
    if isinstance(pooling_type, AvgPooling):
        extra_dict['average_strategy'] = pooling_type.strategy
    elif isinstance(pooling_type, MaxPooling) and \
                    pooling_type.output_max_index is not None:
        assert isinstance(pooling_type.output_max_index, bool)
        extra_dict['output_max_index'] = pooling_type.output_max_index
    extra_dict.update(ExtraLayerAttribute.to_kwargs(layer_attr))

    Layer(
        name=name,
        type=pooling_type.name,
        inputs=[Input(input.name)],
        bias=ParamAttr.to_bias(bias_attr),
        trans_type=agg_level,
        **extra_dict)

    return LayerOutput(
        name, pooling_type.name, parents=[input], size=input.size)


@wrap_bias_attr_default()
@wrap_param_attr_default()
@wrap_act_default(param_names=['gate_act'], act=SigmoidActivation())
@wrap_act_default(param_names=["act", 'state_act'], act=TanhActivation())
@wrap_name_default("lstmemory")
@layer_support(DROPOUT)
def lstmemory(input,
              name=None,
              reverse=False,
              act=None,
              gate_act=None,
              size=None,
              state_act=None,
              bias_attr=None,
              param_attr=None,
              layer_attr=None):
    """
    Long Short-term Memory Cell.

    The memory cell was implemented as follow equations.

    ..  math::

        i_t & = \\sigma(W_{xi}x_{t} + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)

        f_t & = \\sigma(W_{xf}x_{t} + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)

        c_t & = f_tc_{t-1} + i_t tanh (W_{xc}x_t+W_{hc}h_{t-1} + b_c)

        o_t & = \\sigma(W_{xo}x_{t} + W_{ho}h_{t-1} + W_{co}c_t + b_o)

        h_t & = o_t tanh(c_t)


    NOTE: In PaddlePaddle's implementation, the multiplications
    :math:`W_{xi}x_{t}` , :math:`W_{xf}x_{t}`,
    :math:`W_{xc}x_t`, :math:`W_{xo}x_{t}` are not done in the lstmemory layer,
    so an additional mixed_layer with full_matrix_projection or a fc_layer must
    be included in the configuration file to complete the input-to-hidden
    mappings before lstmemory is called.

    NOTE: This is a low level user interface. You can use network.simple_lstm
    to config a simple plain lstm layer.

    Please refer to **Generating Sequences With Recurrent Neural Networks** for
    more details about LSTM.

    Link_ goes as below.

    .. _Link: http://arxiv.org/abs/1308.0850

    :param name: The lstmemory layer name.
    :type name: basestring
    :param input: input layer name.
    :type input: LayerOutput
    :param reverse: is sequence process reversed or not.
    :type reverse: bool
    :param act: activation type, TanhActivation by default. :math:`h_t`
    :type act: BaseActivation
    :param gate_act: gate activation type, SigmoidActivation by default.
    :type gate_act: BaseActivation
    :param state_act: state activation type, TanhActivation by default.
    :type state_act: BaseActivation

    :param bias_attr: Bias attribute. None means default bias. False means no
                      bias.
    :type bias_attr: ParameterAttribute|None|False
    :param param_attr: Parameter Attribute.
    :type param_attr: ParameterAttribute|None|False
    :param layer_attr: Extra Layer attribute
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert gate_act.support_hppl
    assert state_act.support_hppl
    assert act.support_hppl
    assert input.size is not None and input.size % 4 == 0
    if size is not None:
        if input.size / 4 == size:
            plog = logger.warning
        else:
            plog = logger.fatal

        plog("NOTE: The lstmemory layer[%s]'s size is set by previous input "
             "layer. The lstm size should be equal with input layer size/4. The"
             " size which is set explicitly will be ignored." % name)

    Layer(
        name=name,
        type=LayerType.LSTMEMORY,
        active_type=act.name,
        active_state_type=state_act.name,
        active_gate_type=gate_act.name,
        reversed=reverse,
        bias=ParamAttr.to_bias(bias_attr),
        inputs=[Input(input.name, **param_attr.attr)],
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(
        name,
        LayerType.LSTMEMORY, [input],
        size=input.size / 4,
        reverse=reverse)


@wrap_bias_attr_default()
@wrap_param_attr_default()
@wrap_act_default(param_names=['gate_act'], act=SigmoidActivation())
@wrap_act_default(param_names=["act"], act=TanhActivation())
@wrap_name_default("gru")
@layer_support(DROPOUT)
def grumemory(input,
              name=None,
              reverse=False,
              act=None,
              gate_act=None,
              size=None,
              bias_attr=None,
              param_attr=None,
              layer_attr=None):
    """
    Gate Recurrent Unit Layer.

    The memory cell was implemented as follow equations.

    1. update gate :math:`z`: defines how much of the previous memory to
    keep around or the unit updates its activations. The update gate
    is computed by:

    ..  math::

        z_t = \\sigma(W_{z}x_{t} + U_{z}h_{t-1} + b_z)

    2. reset gate :math:`r`: determines how to combine the new input with the
    previous memory. The reset gate is computed similarly to the update gate:

    ..  math::

        r_t = \\sigma(W_{r}x_{t} + U_{r}h_{t-1} + b_r)

    3. The candidate activation :math:`\\tilde{h_t}` is computed similarly to
    that of the traditional recurrent unit:

    ..  math::

        {\\tilde{h_t}} = tanh(W x_{t} + U (r_{t} \odot h_{t-1}) + b)

    4. The hidden activation :math:`h_t` of the GRU at time t is a linear
    interpolation between the previous activation :math:`h_{t-1}` and the
    candidate activation :math:`\\tilde{h_t}`:

    ..  math::

        h_t = (1 - z_t) h_{t-1} + z_t {\\tilde{h_t}}

    NOTE: In PaddlePaddle's implementation, the multiplication operations
    :math:`W_{r}x_{t}`, :math:`W_{z}x_{t}` and :math:`W x_t` are not computed in
    gate_recurrent layer. Consequently, an additional mixed_layer with
    full_matrix_projection or a fc_layer must be included before grumemory
    is called.

    More details can be found by referring to `Empirical Evaluation of Gated
    Recurrent Neural Networks on Sequence Modeling.
    <https://arxiv.org/abs/1412.3555>`_

    The simple usage is:

    .. code-block:: python

       gru = grumemory(input)

    :param name: The gru layer name.
    :type name: None|basestring
    :param input: input layer.
    :type input: LayerOutput.
    :param reverse: Whether sequence process is reversed or not.
    :type reverse: bool
    :param act: activation type, TanhActivation by default. This activation
                affects the :math:`{\\tilde{h_t}}`.
    :type act: BaseActivation
    :param gate_act: gate activation type, SigmoidActivation by default.
                     This activation affects the :math:`z_t` and :math:`r_t`. It is the
                     :math:`\\sigma` in the above formula.
    :type gate_act: BaseActivation
    :param bias_attr: Bias attribute. None means default bias. False means no
                      bias.
    :type bias_attr: ParameterAttribute|None|False
    :param param_attr: Parameter Attribute.
    :type param_attr: ParameterAttribute|None|False
    :param layer_attr: Extra Layer attribute
    :type layer_attr: ExtraLayerAttribute|None
    :param size: Stub parameter of size, but actually not used. If set this size
                 will get a warning.
    :type size: None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert act.support_hppl
    assert gate_act.support_hppl
    assert input.size is not None and input.size % 3 == 0
    if size is not None:
        if input.size / 3 == size:
            plog = logger.warning
        else:
            plog = logger.fatal
        plog("NOTE: the gru memory layer's size is set by previous input layer,"
             " and should be input size / 3. Set size explicitly will be "
             "ignored.")

    Layer(
        name=name,
        type=LayerType.GRUMEMORY,
        active_type=act.name,
        active_gate_type=gate_act.name,
        reversed=reverse,
        bias=ParamAttr.to_bias(bias_attr),
        inputs=[Input(input.name, **param_attr.attr)],
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(
        name,
        LayerType.GRUMEMORY, [input],
        size=input.size / 3,
        reverse=reverse)


@wrap_name_default()
@layer_support()
def last_seq(input,
             name=None,
             agg_level=AggregateLevel.EACH_TIMESTEP,
             stride=-1,
             layer_attr=None):
    """
    Get Last Timestamp Activation of a sequence.

    If stride > 0, this layer slides a window whose size is determined by stride, 
    and return the last value of the window as the output. Thus, a long sequence 
    will be shorten. Note that for sequence with sub-sequence, the default value 
    of stride is -1.

    The simple usage is:

    .. code-block:: python

       seq = last_seq(input=layer)

    :param agg_level: Aggregated level
    :param name: Layer name.
    :type name: basestring
    :param input: Input layer name.
    :type input: LayerOutput
    :param stride: window size.  
    :type stride: Int
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if input.reverse is not None and input.reverse:
        logger.warning("You are getting the last instance of a sequence that"
                       " is a output of a REVERSED layer. There is no time"
                       " series information at all. Maybe you want to use"
                       " first_seq instead.")

    if agg_level == AggregateLevel.EACH_SEQUENCE:
        assert stride == -1

    Layer(
        name=name,
        type=LayerType.SEQUENCE_LAST_INSTANCE,
        inputs=[input.name],
        trans_type=agg_level,
        stride=stride,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        LayerType.SEQUENCE_LAST_INSTANCE,
        parents=[input],
        size=input.size)


@wrap_name_default()
@layer_support()
def first_seq(input,
              name=None,
              agg_level=AggregateLevel.EACH_TIMESTEP,
              stride=-1,
              layer_attr=None):
    """
    Get First Timestamp Activation of a sequence.

    If stride > 0, this layer slides a window whose size is determined by stride, 
    and return the first value of the window as the output. Thus, a long sequence 
    will be shorten. Note that for sequence with sub-sequence, the default value 
    of stride is -1.

    The simple usage is:

    .. code-block:: python

       seq = first_seq(input=layer)

    :param agg_level: aggregation level
    :param name: Layer name.
    :type name: basestring
    :param input: Input layer name.
    :type input: LayerOutput
    :param stride: window size.  
    :type stride: Int
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    if input.reverse is not None and not input.reverse:
        logger.warning('You are getting the first instance for a time series,'
                       ' and it is a normal recurrent layer output. There is no'
                       ' time series information at all. Maybe you want to use'
                       ' last_seq instead.')

    if agg_level == AggregateLevel.EACH_SEQUENCE:
        assert stride == -1

    Layer(
        name=name,
        type=LayerType.SEQUENCE_FIRST_INSTANCE,
        inputs=[input.name],
        trans_type=agg_level,
        stride=stride,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        LayerType.SEQUENCE_FIRST_INSTANCE,
        parents=[input],
        size=input.size)


class ExpandLevel(object):
    FROM_TIMESTEP = AggregateLevel.EACH_TIMESTEP
    FROM_SEQUENCE = AggregateLevel.EACH_SEQUENCE


@wrap_name_default()
@layer_support()
def expand_layer(input,
                 expand_as,
                 name=None,
                 bias_attr=False,
                 expand_level=ExpandLevel.FROM_TIMESTEP,
                 layer_attr=None):
    """
    A layer for "Expand Dense data or (sequence data where the length of each
    sequence is one) to sequence data."

    The example usage is:

    .. code-block:: python

       expand = expand_layer(input=layer1,
                             expand_as=layer2,
                             expand_level=ExpandLevel.FROM_TIMESTEP)

    :param input: Input layer
    :type input: LayerOutput
    :param expand_as: Expand as this layer's sequence info.
    :type expand_as: LayerOutput
    :param name: Layer name.
    :type name: basestring
    :param bias_attr: Bias attribute. None means default bias. False means no
                      bias.
    :type bias_attr: ParameterAttribute|None|False
    :param expand_level: whether input layer is timestep(default) or sequence.
    :type expand_level: ExpandLevel
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    Layer(
        inputs=[input.name, expand_as.name],
        name=name,
        bias=ParamAttr.to_bias(bias_attr=bias_attr),
        type=LayerType.EXPAND_LAYER,
        trans_type=expand_level,
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        size=input.size,
        layer_type=LayerType.EXPAND_LAYER,
        parents=[input, expand_as])


@wrap_name_default()
@layer_support()
def repeat_layer(input, num_repeats, name=None, layer_attr=None):
    """
    A layer for repeating the input for num_repeats times. This is equivalent
    to apply concat_layer() with num_repeats same input.

    .. math::
       y  = [x, x, \cdots, x]

    The example usage is:

    .. code-block:: python

       expand = repeat_layer(input=layer, num_repeats=4)

    :param input: Input layer
    :type input: LayerOutput
    :param num_repeats: Repeat the input so many times
    :type num_repeats: int
    :param name: Layer name.
    :type name: basestring
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    l = Layer(
        inputs=[input.name],
        name=name,
        num_filters=num_repeats,
        type=LayerType.FEATURE_MAP_EXPAND_LAYER,
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        size=l.config.size,
        layer_type=LayerType.FEATURE_MAP_EXPAND_LAYER,
        parents=[input])


@wrap_name_default("seqreshape")
@wrap_act_default(act=IdentityActivation())
@wrap_bias_attr_default(has_bias=False)
@layer_support()
def seq_reshape_layer(input,
                      reshape_size,
                      act=None,
                      name=None,
                      layer_attr=None,
                      bias_attr=None):
    """
    A layer for reshaping the sequence. Assume the input sequence has T instances,
    the dimension of each instance is M, and the input reshape_size is N, then the 
    output sequence has T*M/N instances, the dimension of each instance is N.

    Note that T*M/N must be an integer.

    The example usage is:

    .. code-block:: python

       reshape = seq_reshape_layer(input=layer, reshape_size=4)

    :param input: Input layer.
    :type input: LayerOutput
    :param reshape_size: the size of reshaped sequence.
    :type reshape_size: int
    :param name: Layer name.
    :type name: basestring
    :param act: Activation type.
    :type act: BaseActivation
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :param bias_attr: The Bias Attribute. If no bias, then pass False or
                      something not type of ParameterAttribute. None will get a
                      default Bias.
    :type bias_attr: ParameterAttribute or None or bool
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    Layer(
        inputs=[input.name],
        name=name,
        size=reshape_size,
        type=LayerType.SEQUENCE_RESHAPE,
        bias=ParamAttr.to_bias(bias_attr),
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        size=reshape_size,
        layer_type=LayerType.SEQUENCE_RESHAPE,
        parents=[input])


@wrap_name_default()
@layer_support()
def interpolation_layer(input, weight, name=None, layer_attr=None):
    """
    This layer is for linear interpolation with two inputs,
    which is used in NEURAL TURING MACHINE.

    .. math::
       y.row[i] = w[i] * x_1.row[i] + (1 - w[i]) * x_2.row[i]

    where :math:`x_1` and :math:`x_2` are two (batchSize x dataDim) inputs,
    :math:`w` is (batchSize x 1) weight vector, and :math:`y` is
    (batchSize x dataDim) output.

    The example usage is:

    .. code-block:: python

       interpolation = interpolation_layer(input=[layer1, layer2], weight=layer3)

    :param input: Input layer.
    :type input: list|tuple
    :param weight: Weight layer.
    :type weight: LayerOutput
    :param name: Layer name.
    :type name: basestring
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, collections.Sequence)
    assert len(input) == 2
    assert isinstance(input[0], LayerOutput) and isinstance(input[1],
                                                            LayerOutput)
    if input[0].size is not None and input[1].size is not None:
        assert input[0].size == input[1].size
    assert isinstance(weight, LayerOutput)
    if weight.size is not None:
        assert weight.size == 1
    Layer(
        name=name,
        type=LayerType.INTERPOLATION_LAYER,
        inputs=[weight.name, input[0].name, input[1].name],
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        LayerType.INTERPOLATION_LAYER,
        parents=[weight, input[0], input[1]],
        size=input[0].size)


@wrap_name_default()
@layer_support()
def bilinear_interp_layer(input,
                          out_size_x=None,
                          out_size_y=None,
                          name=None,
                          layer_attr=None):
    """
    This layer is to implement bilinear interpolation on conv layer output.

    Please refer to Wikipedia: https://en.wikipedia.org/wiki/Bilinear_interpolation

    The simple usage is:

    .. code-block:: python

       bilinear = bilinear_interp_layer(input=layer1, out_size_x=64, out_size_y=64)

    :param   input:        A input layer.
    :type    input:        LayerOutput.
    :param   out_size_x:   bilinear interpolation output width.
    :type    out_size_x:   int|None
    :param   out_size_y:   bilinear interpolation output height.
    :type    out_size_y:   int|None
    :param   name:         The layer's name, which cna not be specified.
    :type    name:         None|basestring
    :param   layer_attr:   Extra Layer attribute.
    :type    layer_attr:   ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype:  LayerOutput
    """
    assert input.layer_type == LayerType.CONV_LAYER
    assert isinstance(input.activation, LinearActivation)
    assert out_size_x > 0 and out_size_y > 0
    assert input.num_filters is not None
    num_channels = input.num_filters
    l = Layer(
        name=name,
        inputs=Input(
            input.name,
            bilinear_interp=BilinearInterp(
                out_size_x=out_size_x,
                out_size_y=out_size_y,
                channels=num_channels)),
        type=LayerType.BILINEAR_INTERP_LAYER,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        LayerType.BILINEAR_INTERP_LAYER,
        parents=[input],
        num_filters=num_channels,
        size=l.config.size)


@wrap_name_default()
@layer_support()
def power_layer(input, weight, name=None, layer_attr=None):
    """
    This layer applies a power function to a vector element-wise,
    which is used in NEURAL TURING MACHINE.

    .. math::
       y = x^w

    where :math:`x` is a input vector, :math:`w` is scalar weight,
    and :math:`y` is a output vector.

    The example usage is:

    .. code-block:: python

       power = power_layer(input=layer1, weight=layer2)

    :param input: Input layer.
    :type input: LayerOutput
    :param weight: Weight layer.
    :type weight: LayerOutput
    :param name: Layer name.
    :type name: basestring
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput) and isinstance(weight, LayerOutput)
    if weight.size is not None:
        assert weight.size == 1
    Layer(
        name=name,
        type=LayerType.POWER_LAYER,
        inputs=[weight.name, input.name],
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.POWER_LAYER, parents=[input, weight], size=input.size)


@wrap_name_default()
@layer_support()
def scaling_layer(input, weight, name=None, layer_attr=None):
    """
    A layer for multiplying input vector by weight scalar.

    .. math::
       y  = w x

    where :math:`x` is size=dataDim input, :math:`w` is size=1 weight,
    and :math:`y` is size=dataDim output.

    Note that the above computation is for one sample. Multiple samples are
    processed in one batch.

    The example usage is:

    .. code-block:: python

       scale = scaling_layer(input=layer1, weight=layer2)

    :param input: Input layer.
    :type input: LayerOutput
    :param weight: Weight layer.
    :type weight: LayerOutput
    :param name: Layer name.
    :type name: basestring
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(weight, LayerOutput) and isinstance(input, LayerOutput)
    if weight.size is not None:
        assert weight.size == 1
    Layer(
        name=name,
        type=LayerType.SCALING_LAYER,
        inputs=[weight.name, input.name],
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.SCALING_LAYER, parents=[weight, input], size=input.size)


@wrap_name_default()
@layer_support()
def trans_layer(input, name=None, layer_attr=None):
    """
    A layer for transposing a minibatch matrix.

    .. math::
       y = x^\mathrm{T}

    where :math:`x` is (M x N) input, and :math:`y` is (N x M) output.

    The example usage is:

    .. code-block:: python

       trans = trans_layer(input=layer)

    :param input: Input layer.
    :type input: LayerOutput
    :param name: Layer name.
    :type name: basestring
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    Layer(
        name=name,
        type=LayerType.TRANS_LAYER,
        inputs=[input.name],
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.TRANS_LAYER, parents=[input], size=input.size)


@wrap_name_default()
@layer_support()
def rotate_layer(input, height, width, name=None, layer_attr=None):
    """
    A layer for rotating 90 degrees (clock-wise) for each feature channel,
    usually used when the input sample is some image or feature map.

    .. math::
       y(j,i,:) = x(M-i-1,j,:)

    where :math:`x` is (M x N x C) input, and :math:`y` is (N x M x C) output.

    The example usage is:

    .. code-block:: python

       rot = rotate_layer(input=layer,
                          height=100,
                          width=100)

    :param input: Input layer.
    :type input: LayerOutput
    :param height: The height of the sample matrix
    :type height: int
    :param name: Layer name.
    :type name: basestring
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput)
    l = Layer(
        name=name,
        height=height,
        width=width,
        type=LayerType.ROTATE_LAYER,
        inputs=[input.name],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        layer_type=LayerType.ROTATE_LAYER,
        parents=[input],
        size=l.config.size)


@wrap_name_default()
@layer_support()
def cos_sim(a, b, scale=1, size=1, name=None, layer_attr=None):
    """
    Cosine Similarity Layer. The cosine similarity equation is here.

    ..  math::
        similarity = cos(\\theta) = {\\mathbf{a} \\cdot \\mathbf{b}
        \\over \\|\\mathbf{a}\\| \\|\\mathbf{b}\\|}

    The size of a is M, size of b is M*N,
    Similarity will be calculated N times by step M. The output size is
    N. The scale will be multiplied to similarity.

    Note that the above computation is for one sample. Multiple samples are
    processed in one batch.

    The example usage is:

    .. code-block:: python

       cos = cos_sim(a=layer1, b=layer2, size=3)

    :param name: layer name
    :type name: basestring
    :param a: input layer a
    :type a: LayerOutput
    :param b: input layer b
    :type b: LayerOutput
    :param scale: scale for cosine value. default is 5.
    :type scale: float
    :param size: layer size. NOTE size_a * size should equal size_b.
    :type size: int
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(a, LayerOutput) and isinstance(b, LayerOutput)
    if size == 1:
        Layer(
            name=name,
            type=LayerType.COSINE_SIM,
            cos_scale=scale,
            inputs=[a.name, b.name],
            **ExtraLayerAttribute.to_kwargs(layer_attr))
    else:
        if a.size is not None and b.size is not None:
            assert size == b.size / a.size
        Layer(
            name=name,
            type=LayerType.COSINE_SIM_VEC,
            size=size,
            cos_scale=scale,
            inputs=[a.name, b.name],
            **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(name, LayerType.COSINE_SIM, parents=[a, b], size=size)


@wrap_name_default()
@wrap_bias_attr_default(has_bias=True)
@wrap_param_attr_default()
@layer_support()
def hsigmoid(input,
             label,
             num_classes=None,
             name=None,
             bias_attr=None,
             param_attr=None,
             layer_attr=None):
    """
    Organize the classes into a binary tree. At each node, a sigmoid function
    is used to calculate the probability of belonging to the right branch.
    This idea is from "F. Morin, Y. Bengio (AISTATS 05):
    Hierarchical Probabilistic Neural Network Language Model."

    The example usage is:

    ..  code-block:: python

        cost = hsigmoid(input=[layer1, layer2],
                        label=data_layer)

    :param input: Input layers. It could be a LayerOutput or list/tuple of
                 LayerOutput.
    :type input: LayerOutput|list|tuple
    :param label: Label layer.
    :type label: LayerOutput
    :param num_classes: number of classes.
    :type num_classes: int|None
    :param name: layer name
    :type name: basestring
    :param bias_attr: Bias attribute. None means default bias.
                      False means no bias.
    :type bias_attr: ParameterAttribute|False
    :param param_attr: Parameter Attribute. None means default parameter.
    :type param_attr: ParameterAttribute|None
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if isinstance(input, LayerOutput):
        input = [input]
        if not isinstance(param_attr, collections.Sequence):
            param_attr = [param_attr]
    else:
        if not isinstance(param_attr, collections.Sequence):
            param_attr = [param_attr] * len(input)
        else:
            assert len(param_attr) == len(input)

    assert isinstance(input, collections.Sequence)
    assert isinstance(label, LayerOutput)
    assert label.layer_type == LayerType.DATA

    if num_classes is None:
        num_classes = label.size
    if num_classes is None or num_classes <= 2:
        raise ValueError("hsigmoid label size must larger than 2.")

    ipts_for_layer = []
    parents = []
    for each_input, each_param_attr in zip(input, param_attr):
        assert isinstance(each_input, LayerOutput)
        ipts_for_layer.append(Input(each_input.name, **each_param_attr.attr))
        parents.append(each_input)
    ipts_for_layer.append(label.name)
    parents.append(label)

    l = Layer(
        name=name,
        type=LayerType.HSIGMOID,
        num_classes=num_classes,
        bias=ParamAttr.to_bias(bias_attr),
        inputs=ipts_for_layer,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.HSIGMOID, parents=parents, size=l.config.size)


@wrap_name_default("conv")
@wrap_param_attr_default()
@wrap_bias_attr_default()
@wrap_act_default(act=ReluActivation())
@layer_support(DROPOUT)
def img_conv_layer(input,
                   filter_size,
                   num_filters,
                   name=None,
                   num_channels=None,
                   act=None,
                   groups=1,
                   stride=1,
                   padding=0,
                   bias_attr=None,
                   param_attr=None,
                   shared_biases=True,
                   layer_attr=None,
                   filter_size_y=None,
                   stride_y=None,
                   padding_y=None,
                   trans=False,
                   layer_type=None):
    """
    Convolution layer for image. Paddle can support both square and non-square
    input currently.

    The details of convolution layer, please refer UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/
    FeatureExtractionUsingConvolution/>`_ .

    Convolution Transpose (deconv) layer for image. Paddle can support both square
    and non-square input currently.

    The details of convolution transpose layer,
    please refer to the following explanation and references therein
    <http://datascience.stackexchange.com/questions/6107/
    what-are-deconvolutional-layers/>`_ .
    The num_channel means input image's channel number. It may be 1 or 3 when
    input is raw pixels of image(mono or RGB), or it may be the previous layer's
    num_filters * num_group.

    There are several group of filter in PaddlePaddle implementation.
    Each group will process some channel of the inputs. For example, if an input
    num_channel = 256, group = 4, num_filter=32, the PaddlePaddle will create
    32*4 = 128 filters to process inputs. The channels will be split into 4
    pieces. First 256/4 = 64 channels will process by first 32 filters. The
    rest channels will be processed by rest group of filters.

    The example usage is:

    ..  code-block:: python

        conv = img_conv_layer(input=data, filter_size=1, filter_size_y=1,
                              num_channels=8,
                              num_filters=16, stride=1,
                              bias_attr=False,
                              act=ReluActivation())

    :param name: Layer name.
    :type name: basestring
    :param input: Layer Input.
    :type input: LayerOutput
    :param filter_size: The x dimension of a filter kernel. Or input a tuple for
                        two image dimension.
    :type filter_size: int|tuple|list
    :param filter_size_y: The y dimension of a filter kernel. Since PaddlePaddle
                        currently supports rectangular filters, the filter's
                        shape will be (filter_size, filter_size_y).
    :type filter_size_y: int|None
    :param num_filters: Each filter group's number of filter
    :param act: Activation type. Default is tanh
    :type act: BaseActivation
    :param groups: Group size of filters.
    :type groups: int
    :param stride: The x dimension of the stride. Or input a tuple for two image
                   dimension.
    :type stride: int|tuple|list
    :param stride_y: The y dimension of the stride.
    :type stride_y: int
    :param padding: The x dimension of the padding. Or input a tuple for two
                    image dimension
    :type padding: int|tuple|list
    :param padding_y: The y dimension of the padding.
    :type padding_y: int
    :param bias_attr: Convolution bias attribute. None means default bias.
                      False means no bias.
    :type bias_attr: ParameterAttribute|False
    :param num_channels: number of input channels. If None will be set
                        automatically from previous output.
    :type num_channels: int
    :param param_attr: Convolution param attribute. None means default attribute
    :type param_attr: ParameterAttribute
    :param shared_biases: Is biases will be shared between filters or not.
    :type shared_biases: bool
    :param layer_attr: Layer Extra Attribute.
    :type layer_attr: ExtraLayerAttribute
    :param trans: true if it is a convTransLayer, false if it is a convLayer
    :type trans: bool
    :param layer_type: specify the layer_type, default is None. If trans=True,
                       layer_type has to be "exconvt" or "cudnn_convt", 
                       otherwise layer_type has to be either "exconv" or 
                       "cudnn_conv"
    :type layer_type: String
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if num_channels is None:
        assert input.num_filters is not None
        num_channels = input.num_filters

    if filter_size_y is None:
        if isinstance(filter_size, collections.Sequence):
            assert len(filter_size) == 2
            filter_size, filter_size_y = filter_size
        else:
            filter_size_y = filter_size

    if stride_y is None:
        if isinstance(stride, collections.Sequence):
            assert len(stride) == 2
            stride, stride_y = stride
        else:
            stride_y = stride

    if padding_y is None:
        if isinstance(padding, collections.Sequence):
            assert len(padding) == 2
            padding, padding_y = padding
        else:
            padding_y = padding

    if param_attr.attr.get('initial_smart'):
        # special initial for conv layers.
        init_w = (2.0 / (filter_size**2 * num_channels))**0.5
        param_attr.attr["initial_mean"] = 0.0
        param_attr.attr["initial_std"] = init_w
        param_attr.attr["initial_strategy"] = 0
        param_attr.attr["initial_smart"] = False

    if layer_type:
        if trans:
            assert layer_type in ["exconvt", "cudnn_convt"]
        else:
            assert layer_type in ["exconv", "cudnn_conv"]
        lt = layer_type
    else:
        lt = LayerType.CONVTRANS_LAYER if trans else LayerType.CONV_LAYER

    l = Layer(
        name=name,
        inputs=Input(
            input.name,
            conv=Conv(
                filter_size=filter_size,
                padding=padding,
                stride=stride,
                channels=num_channels,
                groups=groups,
                filter_size_y=filter_size_y,
                padding_y=padding_y,
                stride_y=stride_y),
            **param_attr.attr),
        active_type=act.name,
        num_filters=num_filters,
        bias=ParamAttr.to_bias(bias_attr),
        shared_biases=shared_biases,
        type=lt,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        lt,
        parents=[input],
        activation=act,
        num_filters=num_filters,
        size=l.config.size)


@wrap_name_default("pool")
@layer_support()
def img_pool_layer(input,
                   pool_size,
                   name=None,
                   num_channels=None,
                   pool_type=None,
                   stride=1,
                   padding=0,
                   layer_attr=None,
                   pool_size_y=None,
                   stride_y=None,
                   padding_y=None,
                   ceil_mode=True):
    """
    Image pooling Layer.

    The details of pooling layer, please refer ufldl's pooling_ .

    .. _pooling: http://ufldl.stanford.edu/tutorial/supervised/Pooling/

    - ceil_mode=True:

    ..  math::

        w = 1 + int(ceil(input\_width + 2 * padding - pool\_size) / float(stride))
        h = 1 + int(ceil(input\_height + 2 * padding\_y - pool\_size\_y) / float(stride\_y))

    - ceil_mode=False:

    ..  math::

        w = 1 + int(floor(input\_width + 2 * padding - pool\_size) / float(stride))
        h = 1 + int(floor(input\_height + 2 * padding\_y - pool\_size\_y) / float(stride\_y))

    The example usage is:

    ..  code-block:: python

        maxpool = img_pool_layer(input=conv,
                                 pool_size=3,
                                 pool_size_y=5,
                                 num_channels=8,
                                 stride=1,
                                 stride_y=2,
                                 padding=1,
                                 padding_y=2,
                                 pool_type=MaxPooling())

    :param padding: pooling padding width.
    :type padding: int
    :param padding_y: pooling padding height. It's equal to padding by default.
    :type padding_y: int|None
    :param name: name of pooling layer
    :type name: basestring.
    :param input: layer's input
    :type input: LayerOutput
    :param pool_size: pooling window width
    :type pool_size: int
    :param pool_size_y: pooling window height. It's eaqual to pool_size by default.
    :type pool_size_y: int|None
    :param num_channels: number of input channel.
    :type num_channels: int
    :param pool_type: pooling type. MaxPooling or AvgPooling. Default is
                      MaxPooling.
    :type pool_type: BasePoolingType
    :param stride: stride width of pooling.
    :type stride: int
    :param stride_y: stride height of pooling. It is equal to stride by default.
    :type stride_y: int|None
    :param layer_attr: Extra Layer attribute.
    :type layer_attr: ExtraLayerAttribute
    :param ceil_mode: Wether to use ceil mode to calculate output height and with.
                      Defalut is True. If set false, Otherwise use floor.

    :type ceil_mode: bool
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if num_channels is None:
        assert input.num_filters is not None
        num_channels = input.num_filters

    if pool_type is None:
        pool_type = MaxPooling()
    elif isinstance(pool_type, AvgPooling):
        pool_type.name = 'avg'

    type_name = pool_type.name + '-projection' \
        if (
        isinstance(pool_type, AvgPooling) or isinstance(pool_type, MaxPooling)) \
        else pool_type.name

    pool_size_y = pool_size if pool_size_y is None else pool_size_y
    stride_y = stride if stride_y is None else stride_y
    padding_y = padding if padding_y is None else padding_y

    l = Layer(
        name=name,
        type=LayerType.POOL_LAYER,
        inputs=[
            Input(
                input.name,
                pool=Pool(
                    pool_type=type_name,
                    channels=num_channels,
                    size_x=pool_size,
                    start=None,
                    stride=stride,
                    padding=padding,
                    size_y=pool_size_y,
                    stride_y=stride_y,
                    padding_y=padding_y))
        ],
        ceil_mode=ceil_mode,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        LayerType.POOL_LAYER,
        parents=[input],
        num_filters=num_channels,
        size=l.config.size)


@wrap_name_default("spp")
@layer_support()
def spp_layer(input,
              name=None,
              num_channels=None,
              pool_type=None,
              pyramid_height=None,
              layer_attr=None):
    """
    Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition.
    The details please refer to
    `Kaiming He's paper <https://arxiv.org/abs/1406.4729>`_.

    The example usage is:

    ..  code-block:: python

        spp = spp_layer(input=data, 
                        pyramid_height=2, 
                        num_channels=16, 
                        pool_type=MaxPooling())

    :param name: layer name.
    :type name: basestring
    :param input: layer's input.
    :type input: LayerOutput
    :param num_channels: number of input channel.
    :type num_channels: int
    :param pool_type: Pooling type. MaxPooling or AveragePooling. Default is MaxPooling.
    :type scale: BasePoolingType
    :param pyramid_height: pyramid height.
    :type pyramid_height: int
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if num_channels is None:
        assert input.num_filters is not None
        num_channels = input.num_filters

    if pool_type is None:
        pool_type = MaxPooling()
    elif isinstance(pool_type, AvgPooling):
        pool_type.name = 'avg'

    type_name = pool_type.name
    if (isinstance(pool_type, AvgPooling) or isinstance(pool_type, MaxPooling)):
        type_name += '-projection'

    l = Layer(
        name=name,
        type=LayerType.SPP_LAYER,
        inputs=Input(
            input.name,
            spp=SpatialPyramidPool(
                pool_type=type_name,
                channels=num_channels,
                pyramid_height=pyramid_height)),
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        layer_type=LayerType.SPP_LAYER,
        parents=[input],
        num_filters=num_channels,
        size=l.config.size)


def __img_norm_layer__(name, input, size, norm_type, scale, power, num_channels,
                       blocked, layer_attr):
    if num_channels is None:
        assert input.num_filters is not None
        num_channels = input.num_filters

    l = Layer(
        name=name,
        type=LayerType.NORM_LAYER,
        inputs=Input(
            input.name,
            norm=Norm(
                norm_type=norm_type,
                channels=num_channels,
                size=size,
                scale=scale,
                pow=power,
                blocked=blocked)),
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        layer_type=LayerType.NORM_LAYER,
        parents=[input],
        num_filters=num_channels,
        img_norm_type=norm_type,
        size=l.config.size)


@wrap_name_default("crmnorm")
@layer_support()
def img_cmrnorm_layer(input,
                      size,
                      scale=0.0128,
                      power=0.75,
                      name=None,
                      num_channels=None,
                      layer_attr=None):
    """
    Response normalization across feature maps.
    The details please refer to
    `Alex's paper <http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf>`_.

    The example usage is:

    ..  code-block:: python
    
        norm = img_cmrnorm_layer(input=net, size=5)

    :param name: layer name.
    :type name: None|basestring
    :param input: layer's input.
    :type input: LayerOutput
    :param size: Normalize in number of :math:`size` feature maps.
    :type size: int
    :param scale: The hyper-parameter.
    :type scale: float
    :param power: The hyper-parameter.
    :type power: float
    :param num_channels: input layer's filers number or channels. If
                         num_channels is None, it will be set automatically.
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    return __img_norm_layer__(name, input, size, "cmrnorm-projection", scale,
                              power, num_channels, 0, layer_attr)


@wrap_bias_attr_default()
@wrap_param_attr_default(default_factory=lambda _: ParamAttr(initial_mean=1.0,
                                                             initial_std=0.))
@wrap_act_default(act=ReluActivation())
@wrap_name_default("batch_norm")
@layer_support(DROPOUT)
def batch_norm_layer(input,
                     act=None,
                     name=None,
                     num_channels=None,
                     bias_attr=None,
                     param_attr=None,
                     layer_attr=None,
                     batch_norm_type=None,
                     moving_average_fraction=0.9,
                     use_global_stats=None):
    """
    Batch Normalization Layer. The notation of this layer as follow.

    :math:`x` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    The details of batch normalization please refer to this
    `paper <http://arxiv.org/abs/1502.03167>`_.

    The example usage is:

    ..  code-block:: python
    
        norm = batch_norm_layer(input=net, act=ReluActivation())

    :param name: layer name.
    :type name: basestring
    :param input: batch normalization input. Better be linear activation.
                Because there is an activation inside batch_normalization.
    :type input: LayerOutput
    :param batch_norm_type: We have batch_norm and cudnn_batch_norm. batch_norm
                            supports both CPU and GPU. cudnn_batch_norm requires
                            cuDNN version greater or equal to v4 (>=v4). But
                            cudnn_batch_norm is faster and needs less memory
                            than batch_norm. By default (None), we will
                            automaticly select cudnn_batch_norm for GPU and
                            batch_norm for CPU. Otherwise, select batch norm
                            type based on the specified type. If you use cudnn_batch_norm,
                            we suggested you use latest version, such as v5.1.
    :type batch_norm_type: None|string, None or "batch_norm" or "cudnn_batch_norm"
    :param act: Activation Type. Better be relu. Because batch
                     normalization will normalize input near zero.
    :type act: BaseActivation
    :param num_channels: num of image channels or previous layer's number of
                         filters. None will automatically get from layer's
                         input.
    :type num_channels: int
    :param bias_attr: :math:`\\beta`, better be zero when initialize. So the
                      initial_std=0, initial_mean=1 is best practice.
    :type bias_attr: ParameterAttribute
    :param param_attr: :math:`\\gamma`, better be one when initialize. So the
                       initial_std=0, initial_mean=1 is best practice.
    :type param_attr: ParameterAttribute
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :param use_global_stats: whether use moving mean/variance statistics
                             during testing peroid. If None or True,
                             it will use moving mean/variance statistics during
                             testing. If False, it will use the mean
                             and variance of current batch of test data for
                             testing.
    :type use_global_stats: bool|None.
    :param moving_average_fraction: Factor used in the moving average
                                   computation, referred to as facotr,
                                   :math:`runningMean = newMean*(1-factor)
                                   + runningMean*factor`
    :type moving_average_fraction: float.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if not isinstance(act, ReluActivation):
        logger.log(logging.WARN,
                   "%s is not recommend for batch normalization's activation, "
                   "maybe the relu is better" % act.name)

    if not isinstance(input.activation, LinearActivation):
        logger.log(logging.WARN,
                   "The activation should be inside batch normalization, the "
                   "previous layer's activation may be Linear")

    if num_channels is None:
        if input.num_filters is not None:
            num_channels = input.num_filters
        else:
            num_channels = input.size
    assert (batch_norm_type is None) or (batch_norm_type == "batch_norm") or \
           (batch_norm_type == "cudnn_batch_norm")
    l = Layer(
        name=name,
        inputs=Input(
            input.name, image=Image(channels=num_channels), **param_attr.attr),
        active_type=act.name,
        type=LayerType.BATCH_NORM_LAYER,
        batch_norm_type=batch_norm_type,
        bias=ParamAttr.to_bias(bias_attr),
        moving_average_fraction=moving_average_fraction,
        use_global_stats=use_global_stats,
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(
        name=name,
        layer_type=LayerType.BATCH_NORM_LAYER,
        parents=[input],
        activation=act,
        num_filters=num_channels,
        size=l.config.size)


@wrap_name_default()
@layer_support()
def sum_to_one_norm_layer(input, name=None, layer_attr=None):
    """
    A layer for sum-to-one normalization,
    which is used in NEURAL TURING MACHINE.

    .. math::
       out[i] = \\frac {in[i]} {\sum_{k=1}^N in[k]}

    where :math:`in` is a (batchSize x dataDim) input vector,
    and :math:`out` is a (batchSize x dataDim) output vector.

    The example usage is:

    .. code-block:: python

       sum_to_one_norm = sum_to_one_norm_layer(input=layer)

    :param input: Input layer.
    :type input: LayerOutput
    :param name: Layer name.
    :type name: basestring
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    Layer(
        name=name,
        type=LayerType.SUM_TO_ONE_NORM_LAYER,
        inputs=[input.name],
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.SUM_TO_ONE_NORM_LAYER, parents=[input], size=input.size)


@wrap_name_default("addto")
@wrap_act_default(act=LinearActivation())
@wrap_bias_attr_default(has_bias=False)
@layer_support(DROPOUT)
def addto_layer(input, act=None, name=None, bias_attr=None, layer_attr=None):
    """
    AddtoLayer.

    ..  math::

        y = f(\\sum_{i} x_i + b)

    where :math:`y` is output, :math:`x` is input, :math:`b` is bias,
    and :math:`f` is activation function.

    The example usage is:

    ..  code-block:: python

        addto = addto_layer(input=[layer1, layer2],
                            act=ReluActivation(),
                            bias_attr=False)

    This layer just simply add all input layers together, then activate the sum
    inputs. Each input of this layer should be the same size, which is also the
    output size of this layer.

    There is no weight matrix for each input, because it just a simple add
    operation. If you want a complicated operation before add, please use
    mixed_layer.

    It is a very good way to set dropout outside the layers. Since not all
    PaddlePaddle layer support dropout, you can add an add_to layer, set
    dropout here.
    Please refer to dropout_layer for details.

    :param name: Layer name.
    :type name: basestring
    :param input: Input layers. It could be a LayerOutput or list/tuple of
                 LayerOutput.
    :type input: LayerOutput|list|tuple
    :param act: Activation Type, default is tanh.
    :type act: BaseActivation
    :param bias_attr: Bias attribute. If False, means no bias. None is default
                      bias.
    :type bias_attr: ParameterAttribute|bool
    :param layer_attr: Extra Layer attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    num_filters = None
    if isinstance(input, LayerOutput):
        input = [input]

    assert isinstance(input, collections.Sequence)
    ipts_for_layer = []
    for each_input in input:
        assert isinstance(each_input, LayerOutput)
        ipts_for_layer.append(Input(each_input.name))
        if each_input.num_filters is not None:
            num_filters = each_input.num_filters

    l = Layer(
        name=name,
        type=LayerType.ADDTO_LAYER,
        inputs=ipts_for_layer,
        bias=ParamAttr.to_bias(bias_attr),
        active_type=act.name,
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(
        name,
        LayerType.ADDTO_LAYER,
        parents=input,
        activation=act,
        num_filters=num_filters,
        size=l.config.size)


@wrap_act_default(act=IdentityActivation())
@wrap_name_default("concat")
@layer_support()
def concat_layer(input, act=None, name=None, layer_attr=None, bias_attr=None):
    """
    Concat all input vector into one huge vector.
    Inputs can be list of LayerOutput or list of projection.

    The example usage is:

    ..  code-block:: python

        concat = concat_layer(input=[layer1, layer2])

    :param name: Layer name.
    :type name: basestring
    :param input: input layers or projections
    :type input: list|tuple|collections.Sequence
    :param act: Activation type.
    :type act: BaseActivation
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    if isinstance(input, LayerOutput):
        input = [input]
    elif isinstance(input, Projection):
        input = [input]
    else:
        assert isinstance(input, collections.Sequence)

    def __is_type__(o, tp):
        if not isinstance(o, collections.Sequence):
            if o == tp:
                return True
            elif len(o.__bases__) == 0:
                return False
            else:
                for bs in o.__bases__:
                    if __is_type__(bs, tp):
                        return True
                return False
        else:
            tmp = map(lambda _x: __is_type__(_x, tp), o)
            a = tmp[0]
            for b in tmp[1:]:
                assert a == b
            return a

    def __reduce_concat_type__(a, b):
        assert __is_type__([a, b], Projection) or __is_type__([a, b],
                                                              LayerOutput)
        return a

    is_concat_layer = __is_type__(
        reduce(__reduce_concat_type__, map(type, input)), LayerOutput)

    layer_type = (LayerType.CONCAT_LAYER
                  if is_concat_layer else LayerType.CONCAT_PROJ_LAYER)

    if layer_type == LayerType.CONCAT_LAYER:
        assert not bias_attr

    Layer(
        name=name,
        type=layer_type,
        inputs=[x.name for x in input] if is_concat_layer else input,
        active_type=act.name,
        bias=ParamAttr.to_bias(bias_attr),
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    sz = 0
    for each_input in input:
        if each_input.size is not None:
            sz += each_input.size
        else:
            sz = None
            break

    return LayerOutput(
        name,
        layer_type=layer_type,
        parents=input if is_concat_layer else [x.origin for x in input],
        activation=act,
        size=sz)


@wrap_name_default("seqconcat")
@wrap_act_default(act=IdentityActivation())
@wrap_bias_attr_default(has_bias=False)
@layer_support()
def seq_concat_layer(a, b, act=None, name=None, layer_attr=None,
                     bias_attr=None):
    """
    Concat sequence a with sequence b.

    Inputs: 
      - a = [a1, a2, ..., an]
      - b = [b1, b2, ..., bn]
      - Note that the length of a and b should be the same.
        
    Output: [a1, b1, a2, b2, ..., an, bn]

    The example usage is:

    ..  code-block:: python

        concat = seq_concat_layer(a=layer1, b=layer2)

    :param name: Layer name.
    :type name: basestring
    :param a: input sequence layer
    :type a: LayerOutput
    :param b: input sequence layer
    :type b: LayerOutput
    :param act: Activation type.
    :type act: BaseActivation
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :param bias_attr: The Bias Attribute. If no bias, then pass False or
                      something not type of ParameterAttribute. None will get a
                      default Bias.
    :type bias_attr: ParameterAttribute or None or bool
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(a, LayerOutput) and isinstance(b, LayerOutput)
    assert a.size == b.size
    Layer(
        name=name,
        type=LayerType.SEQUENCE_CONCAT_LAYER,
        inputs=[a.name, b.name],
        active_type=act.name,
        bias=ParamAttr.to_bias(bias_attr),
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(
        name,
        layer_type=LayerType.SEQUENCE_CONCAT_LAYER,
        parents=[a, b],
        activation=act,
        size=a.size)


@wrap_name_default("memory", "memory_name")
def memory(name,
           size,
           memory_name=None,
           is_seq=False,
           boot_layer=None,
           boot_bias=None,
           boot_bias_active_type=None,
           boot_with_const_id=None):
    """
    The memory layers is a layer cross each time step. Reference this output
    as previous time step layer :code:`name` 's output.

    The default memory is zero in first time step, previous time step's
    output in the rest time steps.

    If boot_bias, the first time step value is this bias and
    with activation.

    If boot_with_const_id, then the first time stop is a IndexSlot, the
    Arguments.ids()[0] is this :code:`cost_id`.

    If boot_layer is not null, the memory is just the boot_layer's output.
    Set :code:`is_seq` is true boot layer is sequence.

    The same name layer in recurrent group will set memory on each time
    step.

    .. code-block:: python

       mem = memory(size=256, name='state')
       state = fc_layer(input=mem, size=256, name='state')

    If you do not want to specify the name, you can equivalently use set_input()
    to specify the layer needs to be remembered as the following:

    .. code-block:: python
       mem = memory(size=256)
       state = fc_layer(input=mem, size=256)
       mem.set_input(mem)


    :param name: the name of the layer which this memory remembers.
                 If name is None, user should call set_input() to specify the
                 name of the layer which this memory remembers.
    :type name: basestring
    :param size: size of memory.
    :type size: int
    :param memory_name: the name of the memory.
                        It is ignored when name is provided.
    :type memory_name: basestring
    :param is_seq: is sequence for boot_layer
    :type is_seq: bool
    :param boot_layer: boot layer of memory.
    :type boot_layer: LayerOutput|None
    :param boot_bias: boot layer's bias
    :type boot_bias: ParameterAttribute|None
    :param boot_bias_active_type: boot layer's active type.
    :type boot_bias_active_type: BaseActivation
    :param boot_with_const_id: boot layer's id.
    :type boot_with_const_id: int
    :return: LayerOutput object which is a memory.
    :rtype: LayerOutput
    """
    if boot_bias_active_type is None:
        boot_bias_active_type = LinearActivation()

    assert boot_bias is None or isinstance(boot_bias, ParameterAttribute)
    if isinstance(boot_bias, ParameterAttribute):
        boot_bias = ParamAttr.to_bias(boot_bias)

    assert boot_layer is None or isinstance(boot_layer, LayerOutput)
    if name is not None:
        memory_name = None

    memory_name = Memory(
        name,
        size,
        is_sequence=is_seq,
        boot_layer=boot_layer.name if boot_layer is not None else None,
        boot_bias=boot_bias,
        boot_bias_active_type=boot_bias_active_type.name,
        boot_with_const_id=boot_with_const_id,
        memory_name=memory_name)

    lout = LayerOutput(
        name=memory_name,
        size=size,
        layer_type=LayerType.MEMORY,
        parents=[boot_layer] if boot_layer is not None else None)
    return lout


@wrap_bias_attr_default()
@wrap_act_default(
    param_names=['gate_act', 'state_act'], act=SigmoidActivation())
@wrap_act_default(act=TanhActivation())
@wrap_name_default('lstm_step')
@layer_support()
def lstm_step_layer(input,
                    state,
                    size,
                    act=None,
                    name=None,
                    gate_act=None,
                    state_act=None,
                    bias_attr=None,
                    layer_attr=None):
    """
    LSTM Step Layer. It used in recurrent_group. The lstm equations are shown
    as follow.

    ..  math::

        i_t & = \\sigma(W_{xi}x_{t} + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)

        f_t & = \\sigma(W_{xf}x_{t} + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)

        c_t & = f_tc_{t-1} + i_t tanh (W_{xc}x_t+W_{hc}h_{t-1} + b_c)

        o_t & = \\sigma(W_{xo}x_{t} + W_{ho}h_{t-1} + W_{co}c_t + b_o)

        h_t & = o_t tanh(c_t)


    The input of lstm step is :math:`Wx_t + Wh_{t-1}`, and user should use
    :code:`mixed_layer` and :code:`full_matrix_projection` to calculate these
    input vector.

    The state of lstm step is :math:`c_{t-1}`. And lstm step layer will do

    ..  math::

        i_t = \\sigma(input + W_{ci}c_{t-1} + b_i)

        ...


    This layer contains two outputs. Default output is :math:`h_t`. The other
    output is :math:`o_t`, which name is 'state' and can use
    :code:`get_output_layer` to extract this output.

    :param name: Layer's name.
    :type name: basestring
    :param size: Layer's size. NOTE: lstm layer's size, should be equal as
                 :code:`input.size/4`, and should be equal as
                 :code:`state.size`.
    :type size: int
    :param input: input layer. :math:`Wx_t + Wh_{t-1}`
    :type input: LayerOutput
    :param state: State Layer. :math:`c_{t-1}`
    :type state: LayerOutput
    :param act: Activation type. Default is tanh
    :type act: BaseActivation
    :param gate_act: Gate Activation Type. Default is sigmoid, and should
                          be sigmoid only.
    :type gate_act: BaseActivation
    :param state_act: State Activation Type. Default is sigmoid, and should
                           be sigmoid only.
    :type state_act: BaseActivation
    :param bias_attr: Bias Attribute.
    :type bias_attr: ParameterAttribute
    :param layer_attr: layer's extra attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    Layer(
        name=name,
        type=LayerType.LSTM_STEP_LAYER,
        active_type=act.name,
        active_gate_type=gate_act.name,
        active_state_type=state_act.name,
        bias=ParamAttr.to_bias(bias_attr),
        size=size,
        inputs=[input.name, state.name],
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(
        name=name,
        layer_type=LayerType.LSTM_STEP_LAYER,
        parents=[input, state],
        activation=act,
        size=size,
        outputs=['default', 'state'])


@wrap_bias_attr_default()
@wrap_param_attr_default()
@wrap_act_default(param_names=['gate_act'], act=SigmoidActivation())
@wrap_act_default(act=TanhActivation())
@wrap_name_default('gru_step')
@layer_support()
def gru_step_layer(input,
                   output_mem,
                   size=None,
                   act=None,
                   name=None,
                   gate_act=None,
                   bias_attr=None,
                   param_attr=None,
                   layer_attr=None):
    """

    :param input:
    :type input: LayerOutput
    :param output_mem:
    :param size:
    :param act:
    :param name:
    :param gate_act:
    :param bias_attr:
    :param param_attr: the parameter_attribute for transforming the output_mem
                       from previous step.
    :param layer_attr:
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert input.size % 3 == 0
    if size is None:
        size = input.size / 3
    Layer(
        name=name,
        type=LayerType.GRU_STEP_LAYER,
        # The parameter here is for transforming the output_mem. The input has
        # already been transformed outside this module so it does not need
        # parameter associated with it.
        # The parameter here is instead grouped with input is due to
        # backward model compatibility.
        inputs=[Input(input.name, **param_attr.attr), output_mem.name],
        bias=ParamAttr.to_bias(bias_attr),
        size=size,
        active_type=act.name,
        active_gate_type=gate_act.name,
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        layer_type=LayerType.GRU_STEP_LAYER,
        parents=[input, output_mem],
        size=size,
        activation=act)


@wrap_bias_attr_default()
@wrap_param_attr_default()
@wrap_act_default(param_names=['gate_act'], act=SigmoidActivation())
@wrap_act_default(act=TanhActivation())
@wrap_name_default('gru_step_naive')
@layer_support(ERROR_CLIPPING, DROPOUT)
def gru_step_naive_layer(input,
                         output_mem,
                         size=None,
                         name=None,
                         act=None,
                         gate_act=None,
                         bias_attr=None,
                         param_attr=None,
                         layer_attr=None):
    """
    GRU Step Layer, but using MixedLayer to generate. It support ERROR_CLIPPING
    and DROPOUT.

    :param input:
    :param output_mem:
    :param size:
    :param name:
    :param act:
    :param gate_act:
    :param bias_attr:
    :param param_attr:
    :param layer_attr:
    :return:
    """
    if input.size % 3 != 0:
        raise ValueError("GruStep input size must be divided by 3")
    if size is None:
        size = input.size / 3

    def __gate__(gate_name, offset):
        with mixed_layer(
                name=name + "_" + gate_name,
                size=size,
                layer_attr=layer_attr,
                bias_attr=bias_attr,
                act=gate_act) as gate:
            gate += identity_projection(input=input, offset=offset)
            gate += full_matrix_projection(
                input=output_mem, param_attr=param_attr)
        return gate

    update_gate = __gate__("update", 0)
    reset_gate = __gate__("reset", size)

    with mixed_layer(
            name=name + "_reset_output", bias_attr=False) as reset_output:
        reset_output += dotmul_operator(a=output_mem, b=reset_gate)

    with mixed_layer(
            name=name + "_output_candidate",
            size=size,
            layer_attr=layer_attr,
            bias_attr=bias_attr,
            act=act) as output_candidate:
        output_candidate += identity_projection(input=input, offset=2 * size)
        output_candidate += full_matrix_projection(
            input=reset_output, param_attr=param_attr)

    with mixed_layer(name=name) as output:
        output += identity_projection(output_mem)
        output += dotmul_operator(a=output_mem, b=update_gate, scale=-1.0)
        output += dotmul_operator(a=output_candidate, b=update_gate)

    return output


@wrap_name_default()
@layer_support()
def get_output_layer(input, arg_name, name=None, layer_attr=None):
    """
    Get layer's output by name. In PaddlePaddle, a layer might return multiple
    values, but returns one layer's output. If the user wants to use another
    output besides the default one, please use get_output_layer first to get
    the output from input.

    :param name: Layer's name.
    :type name: basestring
    :param input: get output layer's input. And this layer should contains
                   multiple outputs.
    :type input: LayerOutput
    :param arg_name: Output name from input.
    :type arg_name: basestring
    :param layer_attr: Layer's extra attribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    # GetOutputLayer
    assert arg_name in input.outputs, 'Get Output From an not existed input.' \
                                      ' The get output name is %s, which not' \
                                      ' in %s' % (
                                          arg_name, ",".join(input.outputs))
    Layer(
        name=name,
        type=LayerType.GET_OUTPUT_LAYER,
        inputs=[Input(
            input.name, input_layer_argument=arg_name)],
        size=input.size,
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(
        name=name,
        layer_type=LayerType.GET_OUTPUT_LAYER,
        parents=[input],
        size=input.size)


@wrap_name_default()
@wrap_act_default()
@wrap_bias_attr_default()
@wrap_param_attr_default()
@layer_support()
def recurrent_layer(input,
                    act=None,
                    bias_attr=None,
                    param_attr=None,
                    name=None,
                    reverse=False,
                    layer_attr=None):
    """
    Simple recurrent unit layer. It is just a fully connect layer through both
    time and neural network.

    For each sequence [start, end] it performs the following computation\:

    ..  math::

        out_{i} = act(in_{i})     \\      \\      \\text{for} \\ i = start \\\\
        out_{i} = act(in_{i} + out_{i-1} * W) \\ \\ \\text{for} \\ start < i <= end

    If reversed is true, the order is reversed\:

    ..  math::

        out_{i} = act(in_{i})           \\    \\   \\text{for} \\ i = end  \\\\
        out_{i} = act(in_{i} + out_{i+1} * W) \\ \\ \\text{for} \\ start <= i < end


    :param input: Input Layer
    :type input: LayerOutput
    :param act: activation.
    :type act: BaseActivation
    :param bias_attr: bias attribute.
    :type bias_attr: ParameterAttribute
    :param param_attr: parameter attribute.
    :type param_attr: ParameterAttribute
    :param name: name of the layer
    :type name: basestring
    :param layer_attr: Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    Layer(
        name=name,
        type=LayerType.RECURRENT_LAYER,
        inputs=Input(input.name, **param_attr.attr),
        active_type=act.name,
        bias=ParamAttr.to_bias(bias_attr),
        reversed=reverse,
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        layer_type=LayerType.RECURRENT_LAYER,
        parents=[input],
        size=input.size,
        activation=act,
        reverse=reverse)


class StaticInput(object):
    """
    StaticInput is only used in recurrent_group which defines a read-only memory
    that can be a sequence or non-sequence.
    """

    def __init__(self, input, is_seq=False, size=None):
        assert isinstance(input, LayerOutput)
        self.input = input
        self.is_seq = is_seq
        assert input.size is not None or size is not None
        if size is not None:
            input.size = size


class SubsequenceInput(object):
    """
    Input sequence has sub-sequence, used in recurrent_group.

    The example usage is:

    .. code-block:: python

       input = SubsequenceInput(layer)
    """

    def __init__(self, input):
        assert isinstance(input, LayerOutput)
        assert input.size is not None
        self.input = input


@wrap_name_default("recurrent_group")
def recurrent_group(step,
                    input,
                    reverse=False,
                    name=None,
                    targetInlink=None,
                    is_generating=False):
    """
    Recurrent layer group is an extremely flexible recurrent unit in
    PaddlePaddle. As long as the user defines the calculation done within a
    time step, PaddlePaddle will iterate such a recurrent calculation over
    sequence input. This is extremely usefull for attention based model, or
    Neural Turning Machine like models.

    The basic usage (time steps) is:

    .. code-block:: python

       def step(input):
           output = fc_layer(input=layer,
                             size=1024,
                             act=LinearActivation(),
                             bias_attr=False)
           return output

       group = recurrent_group(input=layer,
                               step=step)

    You can see following configs for further usages:

    - time steps: lstmemory_group, paddle/gserver/tests/sequence_layer_group.conf, \
                  demo/seqToseq/seqToseq_net.py
    - sequence steps: paddle/gserver/tests/sequence_nest_layer_group.conf

    :param step: recurrent one time step function.The input of this function is
                 input of the group. The return of this function will be
                 recurrent group's return value.

                 The recurrent group scatter a sequence into time steps. And
                 for each time step, will invoke step function, and return
                 a time step result. Then gather each time step of output into
                 layer group's output.

    :type step: callable

    :param name: recurrent_group's name.
    :type name: basestring

    :param input: Input links array.

                  LayerOutput will be scattered into time steps.
                  SubsequenceInput will be scattered into sequence steps.
                  StaticInput will be imported to each time step, and doesn't change
                  through time. It's a mechanism to access layer outside step function.

    :type input: LayerOutput|StaticInput|SubsequenceInput|list|tuple

    :param reverse: If reverse is set true, the recurrent unit will process the
                    input sequence in a reverse order.
    :type reverse: bool

    :param targetInlink: the input layer which share info with layer group's output

                         Param input specifies multiple input layers. For
                         SubsequenceInput inputs, config should assign one input
                         layer that share info(the number of sentences and the number
                         of words in each sentence) with all layer group's outputs.
                         targetInlink should be one of the layer group's input.

    :type targetInlink: LayerOutput|SubsequenceInput

    :param is_generating: If is generating, none of input type should be LayerOutput;
                          else, for training or testing, one of the input type must
                          be LayerOutput.

    : type is_generating: bool

    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    model_type('recurrent_nn')

    def is_single_input(x):
        return isinstance(x, LayerOutput) or isinstance(x, StaticInput) \
               or isinstance(x, SubsequenceInput)

    if is_single_input(input):
        input = [input]
    assert isinstance(input, collections.Sequence)

    def is_in_links(x):
        return isinstance(x, LayerOutput) or isinstance(x, SubsequenceInput)

    in_links = filter(is_in_links, input)

    def targetInlink_in_inlinks():
        for inlink in in_links:
            if isinstance(inlink, SubsequenceInput):
                if targetInlink == inlink.input:
                    return True
            elif targetInlink == inlink:
                return True
        return False

    assert (targetInlink == None or targetInlink_in_inlinks())
    targetInlinkName = None if targetInlink == None \
        else targetInlink.name if isinstance(targetInlink, LayerOutput) \
        else targetInlink.input.name

    contains_sub_seq = [False]

    def map_in_links(x):
        if isinstance(x, SubsequenceInput):
            contains_sub_seq[0] = True
            return Link(name=x.input.name, has_subseq=True)
        else:
            return x.name

    RecurrentLayerGroupWithoutOutLinksBegin(
        name=name,
        in_links=map(map_in_links, in_links),
        seq_reversed=reverse,
        target_inlinkname=targetInlinkName)
    in_args = []
    has_LayerOutput = False
    for each_input in input:
        assert is_single_input(each_input)
        if isinstance(each_input, LayerOutput):
            in_args.append(each_input)
            has_LayerOutput = True
        elif isinstance(each_input, SubsequenceInput):
            in_args.append(each_input.input)
            has_LayerOutput = True
        else:
            mem_name = "__%s_memory__" % each_input.input.name
            mem = memory(
                name=mem_name,
                is_seq=each_input.is_seq,
                size=each_input.input.size,
                boot_layer=each_input.input)
            with mixed_layer(
                    name=mem_name,
                    size=each_input.input.size,
                    act=IdentityActivation()) as mix:
                mix += identity_projection(mem)
            in_args.append(mem)

    assert (is_generating != has_LayerOutput)

    layer_outs = step(*in_args)

    if isinstance(layer_outs, LayerOutput):
        layer_outs = [layer_outs]

    for ot in layer_outs:
        assert isinstance(ot, LayerOutput)
        ot.reverse = reverse
        if contains_sub_seq[0]:
            RecurrentLayerGroupSetOutLink(Link(ot.name, has_subseq=True))
        else:
            RecurrentLayerGroupSetOutLink(ot.name)

    RecurrentLayerGroupEnd(name=name)

    if len(layer_outs) == 1:
        return layer_outs[0]
    else:
        return layer_outs


class BaseGeneratedInput(object):
    def __init__(self):
        self.bos_id = None
        self.eos_id = None

    def before_real_step(self):
        raise NotImplementedError()

    def after_real_step(self, *args):
        raise NotImplementedError()


class GeneratedInput(BaseGeneratedInput):
    def after_real_step(self, input):
        return maxid_layer(input=input, name='__beam_search_predict__')

    def before_real_step(self):
        predict_id = memory(
            name='__beam_search_predict__',
            size=self.size,
            boot_with_const_id=self.bos_id)

        trg_emb = embedding_layer(
            input=predict_id,
            size=self.embedding_size,
            param_attr=ParamAttr(name=self.embedding_name))
        return trg_emb

    def __init__(self, size, embedding_name, embedding_size):
        super(GeneratedInput, self).__init__()
        self.size = size
        self.embedding_name = embedding_name
        self.embedding_size = embedding_size


@wrap_name_default()
def maxid_layer(input, name=None, layer_attr=None):
    """
    A layer for finding the id which has the maximal value for each sample.
    The result is stored in output.ids.

    The example usage is:

    .. code-block:: python

       maxid = maxid_layer(input=layer)

    :param input: Input layer name.
    :type input: LayerOutput
    :param name: Layer name.
    :type name: basestring
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert isinstance(input, LayerOutput)
    l = Layer(
        name=name,
        type='maxid',
        inputs=[input.name],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        layer_type=LayerType.MAXID_LAYER,
        parents=[input],
        size=l.config.size)


@wrap_name_default()
def out_prod_layer(input1, input2, name=None, layer_attr=None):
    """
    A layer for computing the outer product of two vectors
    The result is a matrix of size(input1) x size(input2)

    The example usage is:

    .. code-block:: python

       out_prod = out_prod_layer(input1=vec1, input2=vec2)

    :param name: Layer name.
    :type name: basestring
    :param input1: The first input layer name.
    :type input: LayerOutput
    :param input2: The second input layer name.
    :type input2: LayerOutput
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert isinstance(input1, LayerOutput)
    assert isinstance(input2, LayerOutput)
    l = Layer(
        name=name,
        type=LayerType.OUT_PROD_LAYER,
        inputs=[input1.name, input2.name],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        layer_type=LayerType.OUT_PROD_LAYER,
        parents=[input1, input2],
        size=l.config.size)


@wrap_name_default()
def eos_layer(input, eos_id, name=None, layer_attr=None):
    """
    A layer for checking EOS for each sample:
    - output_id = (input_id == conf.eos_id)

    The result is stored in output\_.ids.
    It is used by recurrent layer group.

    The example usage is:

    .. code-block:: python

       eos = eos_layer(input=layer, eos_id=id)

    :param name: Layer name.
    :type name: basestring
    :param input: Input layer name.
    :type input: LayerOutput
    :param eos_id: end id of sequence
    :type eos_id: int
    :param layer_attr: extra layer attributes.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    l = Layer(
        name=name,
        type=LayerType.EOSID_LAYER,
        eos_id=eos_id,
        inputs=[input.name],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        layer_type=LayerType.EOSID_LAYER,
        parents=[input],
        size=l.config.size)


@wrap_name_default()
def beam_search(step,
                input,
                bos_id,
                eos_id,
                beam_size,
                max_length=500,
                name=None,
                num_results_per_sample=None):
    """
    Beam search is a heuristic search algorithm used in sequence generation.
    It explores a graph by expanding the most promising nodes in a limited set
    to maintain tractability.

    The example usage is:

    .. code-block:: python

        def rnn_step(input):
            last_time_step_output = memory(name='rnn', size=512)
            with mixed_layer(size=512, name='rnn') as simple_rnn:
                simple_rnn += full_matrix_projection(input)
                simple_rnn += last_time_step_output
            return simple_rnn

        beam_gen = beam_search(name="decoder",
                               step=rnn_step,
                               input=[StaticInput(encoder_last)],
                               bos_id=0,
                               eos_id=1,
                               beam_size=5)

    Please see the following demo for more details:

    - machine translation : demo/seqToseq/translation/gen.conf \
                            demo/seqToseq/seqToseq_net.py

    :param name: Name of the recurrent unit that generates sequences.
    :type name: base string
    :param step: A callable function that defines the calculation in a time
                 step, and it is applied to sequences with arbitrary length by
                 sharing a same set of weights.

                 You can refer to the first parameter of recurrent_group, or
                 demo/seqToseq/seqToseq_net.py for more details.
    :type step: callable
    :param input: Input data for the recurrent unit
    :type input: list
    :param bos_id: Index of the start symbol in the dictionary. The start symbol
                   is a special token for NLP task, which indicates the
                   beginning of a sequence. In the generation task, the start
                   symbol is essential, since it is used to initialize the RNN
                   internal state.
    :type bos_id: int
    :param eos_id: Index of the end symbol in the dictionary. The end symbol is
                   a special token for NLP task, which indicates the end of a
                   sequence. The generation process will stop once the end
                   symbol is generated, or a pre-defined max iteration number
                   is exceeded.
    :type eos_id: int
    :param max_length: Max generated sequence length.
    :type max_length: int
    :param beam_size: Beam search for sequence generation is an iterative search
                      algorithm. To maintain tractability, every iteration only
                      only stores a predetermined number, called the beam_size,
                      of the most promising next words. The greater the beam
                      size, the fewer candidate words are pruned.
    :type beam_size: int
    :param num_results_per_sample: Number of the generated results per input
                                  sequence. This number must always be less than
                                  beam size.
    :type num_results_per_sample: int
    :return: The generated word index.
    :rtype: LayerOutput
    """

    if num_results_per_sample is None:
        num_results_per_sample = beam_size
    if num_results_per_sample > beam_size:
        logger.warning("num_results_per_sample should be less than beam_size")

    if isinstance(input, StaticInput) or isinstance(input, BaseGeneratedInput):
        input = [input]

    generated_input_index = -1

    real_input = []
    for i, each_input in enumerate(input):
        assert isinstance(each_input, StaticInput) or isinstance(
            each_input, BaseGeneratedInput)
        if isinstance(each_input, BaseGeneratedInput):
            assert generated_input_index == -1
            generated_input_index = i
        else:
            real_input.append(each_input)

    assert generated_input_index != -1

    gipt = input[generated_input_index]
    assert isinstance(gipt, BaseGeneratedInput)

    gipt.bos_id = bos_id
    gipt.eos_id = eos_id

    def __real_step__(*args):
        eos_name = "__%s_eos_layer__" % name
        RecurrentLayerGroupSetGenerator(
            Generator(
                eos_layer_name=eos_name,
                max_num_frames=max_length,
                beam_size=beam_size,
                num_results_per_sample=num_results_per_sample))

        args = list(args)
        args.insert(generated_input_index, gipt.before_real_step())

        predict = gipt.after_real_step(step(*args))

        eos_layer(input=predict, eos_id=eos_id, name=eos_name)

        return predict

    tmp = recurrent_group(
        step=__real_step__,
        input=real_input,
        reverse=False,
        name=name,
        is_generating=True)

    return tmp


def __cost_input__(input, label, weight=None):
    """
    inputs and parents for cost layers.
    """
    ipts = [Input(input.name), Input(label.name)]
    parents = [input, label]
    if weight is not None:
        assert weight.size == 1
        ipts.append(Input(weight.name))
        parents.append(weight)
    return ipts, parents


@wrap_name_default()
@layer_support()
def mse_cost(input, label, weight=None, name=None, layer_attr=None):
    """
    mean squared error cost:

    ..  math::

        \frac{1}{N}\sum_{i=1}^N(t_i-y_i)^2

    :param name: layer name.
    :type name: basestring
    :param input: Network prediction.
    :type input: LayerOutput
    :param label: Data label.
    :type label: LayerOutput
    :param weight: The weight affects the cost, namely the scale of cost.
                   It is an optional argument.
    :type weight: LayerOutput
    :param layer_attr: layer's extra attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    ipts, parents = __cost_input__(input, label, weight)

    Layer(
        inputs=ipts,
        type="square_error",
        name=name,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(name, LayerType.COST, parents=parents, size=1)


regression_cost = mse_cost


@wrap_name_default("cost")
@layer_support()
def classification_cost(input,
                        label,
                        weight=None,
                        name=None,
                        top_k=None,
                        evaluator=classification_error_evaluator,
                        layer_attr=None):
    """
    classification cost Layer.

    :param name: layer name.
    :type name: basestring
    :param input: input layer name. network output.
    :type input: LayerOutput
    :param label: label layer name. data_layer often.
    :type label: LayerOutput
    :param weight: The weight affects the cost, namely the scale of cost.
                   It is an optional argument.
    :type weight: LayerOutput
    :param top_k: number k in top-k error rate
    :type top_k: int
    :param evaluator: Evaluator method.
    :param layer_attr: layer's extra attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert input.layer_type != LayerType.DATA
    assert isinstance(input.activation, SoftmaxActivation)
    assert label.layer_type == LayerType.DATA

    ipts, parents = __cost_input__(input, label, weight)

    Layer(
        name=name,
        type="multi-class-cross-entropy",
        inputs=ipts,
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    def __add_evaluator__(e):
        assert callable(e)
        assert hasattr(e, 'is_evaluator')
        assert isinstance(e.is_evaluator, bool)
        assert e.is_evaluator
        assert hasattr(e, "for_classification")
        assert isinstance(e.for_classification, bool)
        assert e.for_classification

        e(name=e.__name__, input=input, label=label, weight=weight, top_k=top_k)

    if not isinstance(evaluator, collections.Sequence):
        evaluator = [evaluator]

    for each_evaluator in evaluator:
        __add_evaluator__(each_evaluator)

    return LayerOutput(name, LayerType.COST, parents=parents, size=1)


def conv_operator(img,
                  filter,
                  filter_size,
                  num_filters,
                  num_channels=None,
                  stride=1,
                  padding=0,
                  filter_size_y=None,
                  stride_y=None,
                  padding_y=None,
                  trans=False):
    """
    Different from img_conv_layer, conv_op is an Operator, which can be used
    in mixed_layer. And conv_op takes two inputs to perform convolution.
    The first input is the image and the second is filter kernel. It only
    support GPU mode.

    The example usage is:

    .. code-block:: python

       op = conv_operator(img=input1,
                          filter=input2,
                          filter_size=3,
                          num_filters=64,
                          num_channels=64)

    :param img: input image
    :type img: LayerOutput
    :param filter: input filter
    :type filter: LayerOutput
    :param filter_size: The x dimension of a filter kernel.
    :type filter_size: int
    :param filter_size_y: The y dimension of a filter kernel. Since
                        PaddlePaddle now supports rectangular filters,
                        the filter's shape can be (filter_size, filter_size_y).
    :type filter_size_y: int
    :param num_filters: channel of output data.
    :type num_filters: int
    :param num_channels: channel of input data.
    :type num_channels: int
    :param stride: The x dimension of the stride.
    :type stride: int
    :param stride_y: The y dimension of the stride.
    :type stride_y: int
    :param padding: The x dimension of padding.
    :type padding: int
    :param padding_y: The y dimension of padding.
    :type padding_y: int
    :return: A ConvOperator Object.
    :rtype: ConvOperator
    """
    if filter_size_y is None:
        filter_size_y = filter_size
    if stride_y is None:
        stride_y = stride
    if padding_y is None:
        padding_y = padding

    if num_channels is None:
        num_channels = img.num_filters

    assert isinstance(filter, LayerOutput)
    if filter.size is not None:
        filter.size = filter_size * filter_size_y * num_filters * num_channels

    opCls = ConvTransOperator if trans else ConvOperator

    op = opCls(
        input_layer_names=[img.name, filter.name],
        num_filters=num_filters,
        conv_conf=Conv(
            filter_size=filter_size,
            padding=padding,
            stride=stride,
            channels=num_channels,
            filter_size_y=filter_size_y,
            padding_y=padding_y,
            stride_y=stride_y,
            groups=1))

    op.origin = [img, filter]
    return op


@wrap_param_attr_default()
def conv_projection(input,
                    filter_size,
                    num_filters,
                    num_channels=None,
                    stride=1,
                    padding=0,
                    filter_size_y=None,
                    stride_y=None,
                    padding_y=None,
                    groups=1,
                    param_attr=None,
                    trans=False):
    """
    Different from img_conv_layer and conv_op, conv_projection is an Projection,
    which can be used in mixed_layer and conat_layer. It use cudnn to implement
    conv and only support GPU mode.

    The example usage is:

    .. code-block:: python

       proj = conv_projection(input=input1,
                              filter_size=3,
                              num_filters=64,
                              num_channels=64)

    :param input: input layer
    :type input: LayerOutput
    :param filter_size: The x dimension of a filter kernel.
    :type filter_size: int
    :param filter_size_y: The y dimension of a filter kernel. Since
                          PaddlePaddle now supports rectangular filters,
                          the filter's shape can be (filter_size, filter_size_y).
    :type filter_size_y: int
    :param num_filters: channel of output data.
    :type num_filters: int
    :param num_channels: channel of input data.
    :type num_channels: int
    :param stride: The x dimension of the stride.
    :type stride: int
    :param stride_y: The y dimension of the stride.
    :type stride_y: int
    :param padding: The x dimension of padding.
    :type padding: int
    :param padding_y: The y dimension of padding.
    :type padding_y: int
    :param groups: The group number.
    :type groups: int
    :param param_attr: Convolution param attribute. None means default attribute
    :type param_attr: ParameterAttribute
    :param trans: whether it is convTrans or conv
    :type trans: boolean
    :return: A DotMulProjection Object.
    :rtype: DotMulProjection
    """
    if num_channels is None:
        assert input.num_filters is not None
        num_channels = input.num_filters

    if filter_size_y is None:
        if isinstance(filter_size, collections.Sequence):
            assert len(filter_size) == 2
            filter_size, filter_size_y = filter_size
        else:
            filter_size_y = filter_size

    if stride_y is None:
        if isinstance(stride, collections.Sequence):
            assert len(stride) == 2
            stride, stride_y = stride
        else:
            stride_y = stride

    if padding_y is None:
        if isinstance(padding, collections.Sequence):
            assert len(padding) == 2
            padding, padding_y = padding
        else:
            padding_y = padding

    if param_attr.attr.get('initial_smart'):
        # special initial for conv layers.
        init_w = (2.0 / (filter_size**2 * num_channels))**0.5
        param_attr.attr["initial_mean"] = 0.0
        param_attr.attr["initial_std"] = init_w
        param_attr.attr["initial_strategy"] = 0
        param_attr.attr["initial_smart"] = False

    projCls = ConvTransProjection if trans else ConvProjection

    proj = projCls(
        input_layer_name=input.name,
        num_filters=num_filters,
        conv_conf=Conv(
            filter_size=filter_size,
            padding=padding,
            stride=stride,
            channels=num_channels,
            filter_size_y=filter_size_y,
            padding_y=padding_y,
            stride_y=stride_y,
            groups=groups),
        **param_attr.attr)

    proj.origin = input
    return proj


@wrap_name_default("pad")
@layer_support()
def pad_layer(input,
              pad_c=None,
              pad_h=None,
              pad_w=None,
              name=None,
              layer_attr=None):
    """
    This operation pads zeros to the input data according to pad_c,pad_h
    and pad_w. pad_c, pad_h, pad_w specifies the which dimension and size
    of padding. And the input data shape is NCHW.

    For example, pad_c=[2,3] means padding 2 zeros before the
    input data and 3 zeros after the input data in channel dimension.
    pad_h means padding zeros in height dimension. pad_w means padding zeros
    in width dimension.

    For example,

    .. code-block:: python

       input(2,2,2,3)  = [
                           [ [[1,2,3], [3,4,5]],
                             [[2,3,5], [1,6,7]] ],
                           [ [[4,3,1], [1,8,7]],
                             [[3,8,9], [2,3,5]] ]
                         ]

       pad_c=[1,1], pad_h=[0,0], pad_w=[0,0]

       output(2,4,2,3) = [
                           [ [[0,0,0], [0,0,0]],
                             [[1,2,3], [3,4,5]],
                             [[2,3,5], [1,6,7]],
                             [[0,0,0], [0,0,0]] ],
                           [ [[0,0,0], [0,0,0]],
                             [[4,3,1], [1,8,7]],
                             [[3,8,9], [2,3,5]],
                             [[0,0,0], [0,0,0]] ]
                         ]

    The simply usage is:

    .. code-block:: python

       pad = pad_layer(input=ipt,
                       pad_c=[4,4],
                       pad_h=[0,0],
                       pad_w=[2,2])

    :param input: layer's input.
    :type input: LayerOutput
    :param pad_c: padding size in channel dimension.
    :type pad_c: list|None
    :param pad_h: padding size in height dimension.
    :type pad_h: list|None
    :param pad_w: padding size in width dimension.
    :type pad_w: list|None
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :param name: layer name.
    :type name: basestring
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if pad_c is not None:
        assert isinstance(pad_c, collections.Sequence) and len(pad_c) == 2
    else:
        pad_c = [0, 0]

    if pad_h is not None:
        assert isinstance(pad_h, collections.Sequence) and len(pad_h) == 2
    else:
        pad_h = [0, 0]

    if pad_w is not None:
        assert isinstance(pad_w, collections.Sequence) and len(pad_w) == 2
    else:
        pad_w = [0, 0]

    assert input.num_filters is not None
    in_ch = input.num_filters
    out_ch = in_ch + pad_c[0] + pad_c[1]

    l = Layer(
        name=name,
        type=LayerType.PAD_LAYER,
        inputs=Input(
            input.name,
            pad=Pad(
                channels=in_ch,
                pad_c=pad_c,
                pad_h=pad_h,
                pad_w=pad_w, )),
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        layer_type=LayerType.PAD_LAYER,
        parents=[input],
        num_filters=out_ch,
        size=l.config.size)


@wrap_name_default()
@layer_support()
def conv_shift_layer(a, b, name=None, layer_attr=None):
    """
    This layer performs cyclic convolution for two input. For example:
      - a[in]: contains M elements.
      - b[in]: contains N elements (N should be odd).
      - c[out]: contains M elements.

    .. math::

        c[i] = \sum_{j=-(N-1)/2}^{(N-1)/2}a_{i+j} * b_{j}

    In this formular:
     - a's index is computed modulo M. When it is negative, then get item from
       the right side (which is the end of array) to the left.
     - b's index is computed modulo N. When it is negative, then get item from
       the right size (which is the end of array) to the left.

    The example usage is:

    .. code-block:: python

       conv_shift = conv_shift_layer(a=layer1, b=layer2)

    :param name: layer name
    :type name: basestring
    :param a: Input layer a.
    :type a: LayerOutput
    :param b: input layer b.
    :type b: LayerOutput
    :param layer_attr: layer's extra attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(a, LayerOutput) and isinstance(b, LayerOutput)
    assert b.size is None or b.size % 2 == 1  # size of b must be odd.
    Layer(
        name=name,
        type=LayerType.CONV_SHIFT_LAYER,
        inputs=[a.name, b.name],
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(
        name, LayerType.CONV_SHIFT_LAYER, parents=[a, b], size=a.size)


@wrap_name_default()
@wrap_param_attr_default()
@wrap_bias_attr_default()
@wrap_act_default(act=LinearActivation())
@layer_support(ERROR_CLIPPING, DROPOUT)
def tensor_layer(a,
                 b,
                 size,
                 act=None,
                 name=None,
                 param_attr=None,
                 bias_attr=None,
                 layer_attr=None):
    """
    This layer performs tensor operation for two input.
    For example, each sample:

    .. math::
       y_{i} = a * W_{i} * {b^\mathrm{T}}, i=0,1,...,K-1

    In this formular:
      - :math:`a`: the first input contains M elements.
      - :math:`b`: the second input contains N elements.
      - :math:`y_{i}`: the i-th element of y.
      - :math:`W_{i}`: the i-th learned weight, shape if [M, N]
      - :math:`b^\mathrm{T}`: the transpose of :math:`b_{2}`.

    The simple usage is:

    .. code-block:: python

       tensor = tensor_layer(a=layer1, b=layer2, size=1000)

    :param name: layer name
    :type name: basestring
    :param a: Input layer a.
    :type a: LayerOutput
    :param b: input layer b.
    :type b: LayerOutput
    :param size: the layer dimension.
    :type size: int.
    :param act: Activation Type. Default is tanh.
    :type act: BaseActivation
    :param param_attr: The Parameter Attribute.
    :type param_attr: ParameterAttribute
    :param bias_attr: The Bias Attribute. If no bias, then pass False or
                      something not type of ParameterAttribute. None will get a
                      default Bias.
    :type bias_attr: ParameterAttribute|None|Any
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(a, LayerOutput) and isinstance(b, LayerOutput)
    Layer(
        name=name,
        size=size,
        type=LayerType.TENSOR_LAYER,
        active_type=act.name,
        bias=ParamAttr.to_bias(bias_attr),
        inputs=[Input(a.name, **param_attr.attr), Input(b.name)],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.TENSOR_LAYER, parents=[a, b], activation=act, size=size)


@wrap_name_default()
@wrap_param_attr_default()
@wrap_bias_attr_default()
@wrap_act_default()
@layer_support()
def selective_fc_layer(input,
                       size,
                       select=None,
                       act=None,
                       name=None,
                       pass_generation=False,
                       has_selected_colums=True,
                       mul_ratio=0.02,
                       param_attr=None,
                       bias_attr=None,
                       layer_attr=None):
    """
    Selectived fully connected layer. Different from fc_layer, the output
    of this layer maybe sparse. It requires an additional input to indicate
    several selected columns for output. If the selected columns is not
    specified, selective_fc_layer acts exactly like fc_layer.

    The simple usage is:

    .. code-block:: python

       sel_fc = selective_fc_layer(input=input, size=128, act=TanhActivation())

    :param name: The Layer Name.
    :type name: basestring
    :param input: The input layer.
    :type input: LayerOutput|list|tuple
    :param select: The select layer. The output of select layer should be a
                   sparse binary matrix, and treat as the mask of selective fc.
                   If is None, acts exactly like fc_layer.
    :type select: LayerOutput
    :param size: The layer dimension.
    :type size: int
    :param act: Activation Type. Default is tanh.
    :type act: BaseActivation
    :param param_attr: The Parameter Attribute.
    :type param_attr: ParameterAttribute
    :param bias_attr: The Bias Attribute. If no bias, then pass False or
                      something not type of ParameterAttribute. None will get a
                      default Bias.
    :type bias_attr: ParameterAttribute|None|Any
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if isinstance(input, LayerOutput):
        input = [input]
        assert not isinstance(param_attr, collections.Sequence)
        param_attr = [param_attr]
    else:
        if isinstance(param_attr, collections.Sequence):
            assert len(input) == len(param_attr)
        else:
            param_attr = [copy.deepcopy(param_attr) for _ in range(len(input))]

    assert isinstance(input, collections.Sequence)
    assert isinstance(select, LayerOutput)
    if select.size is not None:
        assert select.size == size
    Layer(
        inputs=[
            Input(ipt.name, **attr.attr) for ipt, attr in zip(input, param_attr)
        ] + [select.name],
        name=name,
        type=LayerType.SEL_FC_LAYER,
        size=size,
        bias=ParameterAttribute.to_bias(bias_attr),
        active_type=act.name,
        selective_fc_pass_generation=pass_generation,
        has_selected_colums=has_selected_colums,
        selective_fc_full_mul_ratio=mul_ratio,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        LayerType.SEL_FC_LAYER,
        list(input) + [select],
        activation=act,
        size=size)


@wrap_name_default()
@layer_support()
def sampling_id_layer(input, name=None, layer_attr=None):
    """
    A layer for sampling id from multinomial distribution from the input layer.
    Sampling one id for one sample.

    The simple usage is:

    .. code-block:: python

       samping_id = sampling_id_layer(input=input)

    :param input: The input layer.
    :type input: LayerOutput
    :param name: The Layer Name.
    :type name: basestring
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    l = Layer(
        name=name,
        type=LayerType.SAMPLING_ID_LAYER,
        inputs=[Input(input.name)],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.SAMPLING_ID_LAYER, input, size=l.config.size)


@wrap_name_default()
@layer_support()
def slope_intercept_layer(input,
                          name=None,
                          slope=1.0,
                          intercept=0.0,
                          layer_attr=None):
    """
    This layer for applying a slope and an intercept to the input
    element-wise. There is no activation and weight.

    ..  math::
        y = slope * x + intercept

    The simple usage is:

    .. code-block:: python

       scale = slope_intercept_layer(input=input, slope=-1.0, intercept=1.0)

    :param input: The input layer.
    :type input: LayerOutput
    :param name: The Layer Name.
    :type name: basestring
    :param slope: the scale factor.
    :type slope: float.
    :param intercept: the offset.
    :type intercept: float.
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    Layer(
        name=name,
        type=LayerType.SLOPE_INTERCEPT_LAYER,
        slope=slope,
        intercept=intercept,
        inputs=[Input(input.name)],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.SLOPE_INTERCEPT_LAYER, input, size=input.size)


@wrap_name_default()
@layer_support()
def linear_comb_layer(weights, vectors, size=None, name=None, layer_attr=None):
    """
    A layer for weighted sum of vectors takes two inputs.
      - Input: size of weights is M
               size of vectors is M*N
      - Output: a vector of size=N

    .. math::

       z(i) = \sum_{j=0}^{M-1} x(j) y(i+Nj)

    where :math:`0 \le i \le N-1`

    Or in the matrix notation:

    .. math::

       z = x^\mathrm{T} Y

    In this formular:
      - :math:`x`: weights
      - :math:`y`: vectors.
      - :math:`z`: the output.

    Note that the above computation is for one sample. Multiple samples are
    processed in one batch.

    The simple usage is:

    .. code-block:: python

       linear_comb = linear_comb_layer(weights=weight, vectors=vectors,
                                       size=elem_dim)

    :param weights: The weight layer.
    :type weights: LayerOutput
    :param vectors: The vector layer.
    :type vectors: LayerOutput
    :param size: the dimension of this layer.
    :type size: int
    :param name: The Layer Name.
    :type name: basestring
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(weights, LayerOutput) and isinstance(vectors, LayerOutput)
    if vectors.size is not None and weights.size is not None:
        assert vectors.size % weights.size == 0
        if size is None:
            size = vectors.size / weights.size
        else:
            assert size == vectors.size / weights.size
    Layer(
        name=name,
        type=LayerType.LINEAR_COMBINATION_LAYER,
        size=size,
        inputs=[Input(weights.name), Input(vectors.name)],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.LINEAR_COMBINATION_LAYER, [weights, vectors], size=size)


convex_comb_layer = linear_comb_layer


@wrap_name_default()
@layer_support()
def block_expand_layer(input,
                       block_x=0,
                       block_y=0,
                       stride_x=0,
                       stride_y=0,
                       padding_x=0,
                       padding_y=0,
                       num_channels=None,
                       name=None,
                       layer_attr=None):
    """
    Expand feature map to minibatch matrix.
       - matrix width is: block_y * block_x * num_channels
       - matirx height is: outputH * outputW

    .. math::

       outputH = 1 + (2 * padding_y + imgSizeH - block_y + stride_y - 1) / stride_y

       outputW = 1 + (2 * padding_x + imgSizeW - block_x + stride_x - 1) / stride_x

    The expand method is the same with ExpandConvLayer, but saved the transposed
    value. After expanding, output.sequenceStartPositions will store timeline.
    The number of time steps are outputH * outputW and the dimension of each
    time step is block_y * block_x * num_channels. This layer can be used after
    convolution neural network, and before recurrent neural network.

    The simple usage is:

    .. code-block:: python

       block_expand = block_expand_layer(input=layer,
                                         num_channels=128,
                                         stride_x=1,
                                         stride_y=1,
                                         block_x=1,
                                         block_x=3)

    :param input: The input layer.
    :type input: LayerOutput
    :param num_channels: The channel number of input layer.
    :type num_channels: int|None
    :param block_x: The width of sub block.
    :type block_x: int
    :param block_y: The width of sub block.
    :type block_y: int
    :param stride_x: The stride size in horizontal direction.
    :type stride_x: int
    :param stride_y: The stride size in vertical direction.
    :type stride_y: int
    :param padding_x: The padding size in horizontal direction.
    :type padding_x: int
    :param padding_y: The padding size in vertical direction.
    :type padding_y: int
    :param name: The name of this layer, which can not specify.
    :type name: None|basestring.
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if num_channels is None:
        assert input.num_filters is not None
        num_channels = input.num_filters
    l = Layer(
        name=name,
        inputs=Input(
            input.name,
            block_expand=BlockExpand(
                channels=num_channels,
                block_x=block_x,
                block_y=block_y,
                stride_x=stride_x,
                stride_y=stride_y,
                padding_x=padding_x,
                padding_y=padding_y)),
        type=LayerType.BLOCK_EXPAND,
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(
        name, LayerType.BLOCK_EXPAND, parents=[input], size=l.config.size)


@wrap_name_default()
@layer_support()
def maxout_layer(input, groups, num_channels=None, name=None, layer_attr=None):
    """
    A layer to do max out on conv layer output.
      - Input: output of a conv layer.
      - Output: feature map size same as input. Channel is (input channel) / groups.

    So groups should be larger than 1, and the num of channels should be able
    to devided by groups.

    Please refer to Paper:
      - Maxout Networks: http://www.jmlr.org/proceedings/papers/v28/goodfellow13.pdf
      - Multi-digit Number Recognition from Street View \
        Imagery using Deep Convolutional Neural Networks: \
        https://arxiv.org/pdf/1312.6082v4.pdf

    The simple usage is:

    .. code-block:: python

       maxout = maxout_layer(input,
                             num_channels=128,
                             groups=4)

    :param input: The input layer.
    :type input: LayerOutput
    :param num_channels: The channel number of input layer. If None will be set
                     automatically from previous output.
    :type num_channels: int|None
    :param groups: The group number of input layer.
    :type groups: int
    :param name: The name of this layer, which can not specify.
    :type name: None|basestring.
    :param layer_attr: Extra Layer attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert input.layer_type == LayerType.CONV_LAYER
    assert isinstance(input.activation, LinearActivation)
    assert groups > 1
    if num_channels is None:
        assert input.num_filters is not None
        num_channels = input.num_filters
    assert num_channels % groups == 0
    l = Layer(
        name=name,
        inputs=Input(
            input.name, maxout=MaxOut(
                channels=num_channels, groups=groups)),
        type=LayerType.MAXOUT,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.MAXOUT, parents=[input], size=l.config.size)


@wrap_name_default()
@layer_support()
def ctc_layer(input,
              label,
              size=None,
              name=None,
              norm_by_times=False,
              layer_attr=None):
    """
    Connectionist Temporal Classification (CTC) is designed for temporal
    classication task. That is, for sequence labeling problems where the
    alignment between the inputs and the target labels is unknown.

    More details can be found by referring to `Connectionist Temporal
    Classification: Labelling Unsegmented Sequence Data with Recurrent
    Neural Networks <http://machinelearning.wustl.edu/mlpapers/paper_files/
    icml2006_GravesFGS06.pdf>`_

    Note:
        Considering the 'blank' label needed by CTC, you need to use
        (num_classes + 1) as the input size. num_classes is the category number.
        And the 'blank' is the last category index. So the size of 'input' layer, such as
        fc_layer with softmax activation, should be num_classes + 1. The size of ctc_layer
        should also be num_classes + 1.

    The simple usage:

    .. code-block:: python

      ctc = ctc_layer(input=input,
                      label=label,
                      size=9055,
                      norm_by_times=True)

    :param input: The input layer.
    :type input: LayerOutput
    :param label: The data layer of label with variable length.
    :type label: LayerOutput
    :param size: category numbers + 1.
    :type size: int
    :param name: The name of this layer
    :type name: basestring|None
    :param norm_by_times: Whether to normalization by times. False by default.
    :type norm_by_times: bool
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput)
    assert isinstance(label, LayerOutput)
    if label.size is not None:
        if size is not None:
            assert size == label.size + 1
        else:
            size = label.size + 1
    Layer(
        name=name,
        type=LayerType.CTC_LAYER,
        size=size,
        norm_by_times=norm_by_times,
        inputs=[input.name, label.name],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(name, LayerType.CTC_LAYER, [input, label], size=size)


@wrap_name_default()
@layer_support()
def warp_ctc_layer(input,
                   label,
                   size=None,
                   name=None,
                   blank=0,
                   norm_by_times=False,
                   layer_attr=None):
    """
    A layer intergrating the open-source `warp-ctc
    <https://github.com/baidu-research/warp-ctc>` library, which is used in
    `Deep Speech 2: End-toEnd Speech Recognition in English and Mandarin
    <https://arxiv.org/pdf/1512.02595v1.pdf>`, to compute Connectionist Temporal
    Classification (CTC) loss.

    More details of CTC can be found by referring to `Connectionist Temporal
    Classification: Labelling Unsegmented Sequence Data with Recurrent
    Neural Networks <http://machinelearning.wustl.edu/mlpapers/paper_files/
    icml2006_GravesFGS06.pdf>`_

    Note:
        - Let num_classes represent the category number. Considering the 'blank'
          label needed by CTC, you need to use (num_classes + 1) as the input
          size. Thus, the size of both warp_ctc_layer and 'input' layer should
          be set to num_classes + 1.
        - You can set 'blank' to any value ranged in [0, num_classes], which
          should be consistent as that used in your labels.
        - As a native 'softmax' activation is interated to the warp-ctc library,
          'linear' activation is expected instead in the 'input' layer.

    The simple usage:

    .. code-block:: python

      ctc = warp_ctc_layer(input=input,
                           label=label,
                           size=1001,
                           blank=1000,
                           norm_by_times=False)

    :param input: The input layer.
    :type input: LayerOutput
    :param label: The data layer of label with variable length.
    :type label: LayerOutput
    :param size: category numbers + 1.
    :type size: int
    :param name: The name of this layer, which can not specify.
    :type name: basestring|None
    :param blank: the 'blank' label used in ctc
    :type blank: int
    :param norm_by_times: Whether to normalization by times. False by default.
    :type norm_by_times: bool
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput)
    assert isinstance(label, LayerOutput)
    if label.size is not None:
        if size is not None:
            assert size == label.size + 1
        else:
            size = label.size + 1
    Layer(
        name=name,
        type=LayerType.WARP_CTC_LAYER,
        size=size,
        blank=blank,
        norm_by_times=norm_by_times,
        inputs=[input.name, label.name],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.WARP_CTC_LAYER, parents=[input, label], size=size)


@wrap_name_default()
@wrap_param_attr_default()
@layer_support()
def crf_layer(input,
              label,
              size=None,
              weight=None,
              param_attr=None,
              name=None,
              layer_attr=None):
    """
    A layer for calculating the cost of sequential conditional random
    field model.

    The simple usage:

    .. code-block:: python

      crf = crf_layer(input=input,
                      label=label,
                      size=label_dim)

    :param input: The first input layer is the feature.
    :type input: LayerOutput
    :param label: The second input layer is label.
    :type label: LayerOutput
    :param size: The category number.
    :type size: int
    :param weight: The third layer is "weight" of each sample, which is an
                  optional argument.
    :type weight: LayerOutput
    :param param_attr: Parameter attribute. None means default attribute
    :type param_attr: ParameterAttribute
    :param name: The name of this layers. It is not necessary.
    :type name: None|basestring
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput)
    assert isinstance(label, LayerOutput)
    assert weight is None or isinstance(weight, LayerOutput)
    if input.size is not None and label.size is not None:
        assert input.size == label.size
        if size is None:
            size = input.size
        else:
            assert size == input.size

    ipts = [Input(input.name, **param_attr.attr), Input(label.name)]
    if weight is not None:
        ipts.append(Input(weight.name))

    Layer(
        name=name,
        type=LayerType.CRF_LAYER,
        size=size,
        inputs=ipts,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    parents = [input, label]
    if weight is not None:
        parents.append(weight)
    # The size for LayerOutput means the dimension of the output.
    # It's different from the meaning of crf layer, which is the number of
    # classes.
    return LayerOutput(name, LayerType.CRF_LAYER, parents, size=1)


@wrap_name_default()
@wrap_param_attr_default()
@layer_support()
def crf_decoding_layer(input,
                       size,
                       label=None,
                       param_attr=None,
                       name=None,
                       layer_attr=None):
    """
    A layer for calculating the decoding sequence of sequential conditional
    random field model. The decoding sequence is stored in output.ids.
    If a second input is provided, it is treated as the ground-truth label, and
    this layer will also calculate error. output.value[i] is 1 for incorrect
    decoding or 0 for correct decoding.

    The simple usage:

    .. code-block:: python

      crf_decoding = crf_decoding_layer(input=input,
                                        size=label_dim)

    :param input: The first input layer.
    :type input: LayerOutput
    :param size: size of this layer.
    :type size: int
    :param label: None or ground-truth label.
    :type label: LayerOutput or None
    :param param_attr: Parameter attribute. None means default attribute
    :type param_attr: ParameterAttribute
    :param name: The name of this layers. It is not necessary.
    :type name: None|basestring
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert isinstance(input, LayerOutput)
    assert label is None or isinstance(label, LayerOutput)

    ipts = [Input(input.name, **param_attr.attr)]
    if label is not None:
        ipts.append(Input(label.name))

    Layer(
        name=name,
        type=LayerType.CRF_DECODING_LAYER,
        size=size,
        inputs=ipts,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    parents = [input]
    if label is not None:
        parents.append(label)
    # The size for LayerOutput means the dimension of the output.
    # It's different from the meaning of crf layer, which is the number of
    # classes.
    return LayerOutput(name, LayerType.CRF_DECODING_LAYER, parents, size=1)


@wrap_act_default(act=SigmoidActivation())
@wrap_bias_attr_default(has_bias=True)
@wrap_name_default()
@layer_support()
def nce_layer(input,
              label,
              num_classes,
              act=None,
              weight=None,
              num_neg_samples=10,
              neg_distribution=None,
              name=None,
              bias_attr=None,
              layer_attr=None):
    """
    Noise-contrastive estimation.
    Implements the method in the following paper:
    A fast and simple algorithm for training neural probabilistic language models.

    The example usage is:

    .. code-block:: python

       cost = nce_layer(input=layer1, label=layer2, weight=layer3,
                        num_classes=3, neg_distribution=[0.1,0.3,0.6])

    :param name: layer name
    :type name: basestring
    :param input: input layers. It could be a LayerOutput of list/tuple of LayerOutput.
    :type input: LayerOutput|list|tuple|collections.Sequence
    :param label: label layer
    :type label: LayerOutput
    :param weight: weight layer, can be None(default)
    :type weight: LayerOutput
    :param num_classes: number of classes.
    :type num_classes: int
    :param act: Activation, default is Sigmoid.
    :type act: BaseActivation
    :param num_neg_samples: number of negative samples. Default is 10.
    :type num_neg_samples: int
    :param neg_distribution: The distribution for generating the random negative labels.
                             A uniform distribution will be used if not provided.
                             If not None, its length must be equal to num_classes.
    :type neg_distribution: list|tuple|collections.Sequence|None
    :param bias_attr: Bias parameter attribute. True if no bias.
    :type bias_attr: ParameterAttribute|None|False
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: layer name.
    :rtype: LayerOutput
    """
    if isinstance(input, LayerOutput):
        input = [input]
    assert isinstance(input, collections.Sequence)
    assert isinstance(label, LayerOutput)
    assert label.layer_type == LayerType.DATA
    if neg_distribution is not None:
        assert isinstance(neg_distribution, collections.Sequence)
        assert len(neg_distribution) == num_classes
        assert abs(sum(neg_distribution) - 1.0) < 1e-5
    if not isinstance(act, BaseActivation):
        raise TypeError()

    ipts_for_layer = []
    parents = []
    for each_input in input:
        assert isinstance(each_input, LayerOutput)
        ipts_for_layer.append(each_input.name)
        parents.append(each_input)
    ipts_for_layer.append(label.name)
    parents.append(label)

    if weight is not None:
        assert isinstance(weight, LayerOutput)
        assert weight.layer_type == LayerType.DATA
        ipts_for_layer.append(weight.name)
        parents.append(weight)

    l = Layer(
        name=name,
        type=LayerType.NCE_LAYER,
        num_classes=num_classes,
        neg_sampling_dist=neg_distribution,
        active_type=act.name,
        num_neg_samples=num_neg_samples,
        inputs=ipts_for_layer,
        bias=ParamAttr.to_bias(bias_attr),
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        LayerType.NCE_LAYER,
        parents=parents,
        size=l.config.size,
        activation=act)


"""
following are cost Layers.
"""


@wrap_name_default()
@layer_support()
def rank_cost(left,
              right,
              label,
              weight=None,
              name=None,
              coeff=1.0,
              layer_attr=None):
    """
    A cost Layer for learning to rank using gradient descent. Details can refer
    to `papers <http://research.microsoft.com/en-us/um/people/cburges/papers/
    ICML_ranking.pdf>`_.
    This layer contains at least three inputs. The weight is an optional
    argument, which affects the cost.

    .. math::

       C_{i,j} & = -\\tilde{P_{ij}} * o_{i,j} + log(1 + e^{o_{i,j}})

       o_{i,j} & =  o_i - o_j

       \\tilde{P_{i,j}} & = \\{0, 0.5, 1\\} \ or \ \\{0, 1\\}

    In this formula:
      - :math:`C_{i,j}` is the cross entropy cost.
      - :math:`\\tilde{P_{i,j}}` is the label. 1 means positive order
        and 0 means reverse order.
      - :math:`o_i` and :math:`o_j`: the left output and right output.
        Their dimension is one.

    The simple usage:

    .. code-block:: python

      cost = rank_cost(left=out_left,
                       right=out_right,
                       label=label)

    :param left: The first input, the size of this layer is 1.
    :type left: LayerOutput
    :param right: The right input, the size of this layer is 1.
    :type right: LayerOutput
    :param label: Label is 1 or 0, means positive order and reverse order.
    :type label: LayerOutput
    :param weight: The weight affects the cost, namely the scale of cost.
                   It is an optional argument.
    :type weight: LayerOutput
    :param name: The name of this layers. It is not necessary.
    :type name: None|basestring
    :param coeff: The coefficient affects the gradient in the backward.
    :type coeff: float
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert left.size == 1
    assert right.size == 1
    assert label.size == 1

    ipts = [left.name, right.name, label.name]
    parents = [left, right, label]
    if weight is not None:
        ipts.append(weight.name)
        parents.append(weight)

    Layer(
        name=name,
        type=LayerType.RANK_COST,
        inputs=ipts,
        coeff=coeff,
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(name, LayerType.RANK_COST, parents=parents, size=1)


@wrap_name_default()
@layer_support()
def lambda_cost(input,
                score,
                name,
                NDCG_num=5,
                max_sort_size=-1,
                layer_attr=None):
    """
    lambdaCost for lambdaRank LTR approach.

    The simple usage:

    .. code-block:: python

      cost = lambda_cost(input=input,
                         score=score,
                         NDCG_num=8,
                         max_sort_size=-1)

    :param input: Samples of the same query should be loaded as sequence.
    :type input: LayerOutput
    :param score: The 2nd input. Score of each sample.
    :type input: LayerOutput
    :param NDCG_num: The size of NDCG (Normalized Discounted Cumulative Gain),
                     e.g., 5 for NDCG@5. It must be less than for equal to the
                     minimum size of lists.
    :type NDCG_num: int
    :param max_sort_size: The size of partial sorting in calculating gradient.
                          If max_sort_size = -1, then for each list, the
                          algorithm will sort the entire list to get gradient.
                          In other cases, max_sort_size must be greater than or
                          equal to NDCG_num. And if max_sort_size is greater
                          than the size of a list, the algorithm will sort the
                          entire list of get gradient.
    :type max_sort_size: int
    :param name: The name of this layers. It is not necessary.
    :type name: None|basestring
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput) and isinstance(score, LayerOutput)
    if score.size is not None:
        assert score.size == 1
    Layer(
        name=name,
        type=LayerType.LAMBDA_COST,
        inputs=[input.name, score.name],
        NDCG_num=NDCG_num,
        max_sort_size=max_sort_size,
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(
        name, LayerType.LAMBDA_COST, parents=[input, score], size=1)


@wrap_name_default()
@layer_support()
def cross_entropy(input,
                  label,
                  name=None,
                  coeff=1.0,
                  weight=None,
                  layer_attr=None):
    """
    A loss layer for multi class entropy.

    .. code-block:: python

       cost = cross_entropy(input=input_layer,
                            label=label_layer)

    :param input: The first input layer.
    :type input: LayerOutput.
    :param label: The input label.
    :type input: LayerOutput.
    :param name: The name of this layers. It is not necessary.
    :type name: None|basestring.
    :param coeff: The cost is multiplied with coeff.
                  The coefficient affects the gradient in the backward.
    :type coeff: float.
    :param weight: The cost of each sample is multiplied with each weight.
                   The weight should be a layer with size=1. Note that gradient
                   will not be calculated for weight.
    :type weight: LayerOutout
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput.
    """

    ipts, parents = __cost_input__(input, label, weight)
    Layer(
        name=name,
        type=LayerType.CROSS_ENTROPY,
        inputs=ipts,
        coeff=coeff,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(name, LayerType.CROSS_ENTROPY, parents=parents, size=1)


@wrap_name_default()
@layer_support()
def cross_entropy_with_selfnorm(input,
                                label,
                                name=None,
                                coeff=1.0,
                                softmax_selfnorm_alpha=0.1,
                                layer_attr=None):
    """
    A loss layer for multi class entropy with selfnorm.
    Input should be a vector of positive numbers, without normalization.

    .. code-block:: python

       cost = cross_entropy_with_selfnorm(input=input_layer,
                                          label=label_layer)

    :param input: The first input layer.
    :type input: LayerOutput.
    :param label: The input label.
    :type input: LayerOutput.
    :param name: The name of this layers. It is not necessary.
    :type name: None|basestring.
    :param coeff: The coefficient affects the gradient in the backward.
    :type coeff: float.
    :param softmax_selfnorm_alpha: The scale factor affects the cost.
    :type softmax_selfnorm_alpha: float.
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput.
    """
    Layer(
        name=name,
        type=LayerType.CROSS_ENTROPY_WITH_SELFNORM,
        inputs=[input.name, label.name],
        coeff=coeff,
        softmax_selfnorm_alpha=softmax_selfnorm_alpha,
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(
        name,
        LayerType.CROSS_ENTROPY_WITH_SELFNORM,
        parents=[input, label],
        size=1)


@wrap_name_default()
@layer_support()
def sum_cost(input, name=None, layer_attr=None):
    """
    A loss layer which calculate the sum of the input as loss

    .. code-block:: python

       cost = sum_cost(input=input_layer)

    :param input: The first input layer.
    :type input: LayerOutput.
    :param name: The name of this layers. It is not necessary.
    :type name: None|basestring.
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput.
    """
    assert isinstance(input, LayerOutput)
    Layer(
        name=name,
        type=LayerType.SUM_COST,
        inputs=[input.name],
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    return LayerOutput(name, LayerType.SUM_COST, parents=[input], size=1)


@wrap_name_default()
@layer_support()
def huber_cost(input, label, name=None, coeff=1.0, layer_attr=None):
    """
    A loss layer for huber loss.

    .. code-block:: python

       cost = huber_cost(input=input_layer,
                         label=label_layer)

    :param input: The first input layer.
    :type input: LayerOutput.
    :param label: The input label.
    :type input: LayerOutput.
    :param name: The name of this layers. It is not necessary.
    :type name: None|basestring.
    :param coeff: The coefficient affects the gradient in the backward.
    :type coeff: float.
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput.
    """
    assert isinstance(input, LayerOutput)
    if input.size is not None:
        assert input.size == 1
    Layer(
        name=name,
        type=LayerType.HUBER,
        inputs=[input.name, label.name],
        coeff=coeff,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(name, LayerType.HUBER, parents=[input, label], size=1)


@wrap_name_default()
@layer_support()
def multi_binary_label_cross_entropy(input,
                                     label,
                                     name=None,
                                     coeff=1.0,
                                     layer_attr=None):
    """
    A loss layer for multi binary label cross entropy.

    .. code-block:: python

       cost = multi_binary_label_cross_entropy(input=input_layer,
                                               label=label_layer)

    :param input: The first input layer.
    :type input: LayerOutput
    :param label: The input label.
    :type input: LayerOutput
    :param type: The type of cost.
    :type type: basestring
    :param name: The name of this layers. It is not necessary.
    :type name: None|basestring
    :param coeff: The coefficient affects the gradient in the backward.
    :type coeff: float
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    if input.activation is None or \
            not isinstance(input.activation, SigmoidActivation):
        logger.log(
            logging.WARN,
            "%s is not recommend for multi_binary_label_cross_entropy's activation, "
            "maybe the sigmoid is better" % repr(input.activation))

    Layer(
        name=name,
        type=LayerType.MULTI_BIN_LABEL_CROSS_ENTROPY,
        inputs=[input.name, label.name],
        coeff=coeff,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        LayerType.MULTI_BIN_LABEL_CROSS_ENTROPY,
        parents=[input, label],
        size=1)
