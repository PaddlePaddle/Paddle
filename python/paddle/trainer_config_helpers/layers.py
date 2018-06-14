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

import paddle.trainer.config_parser as cp
from paddle.trainer.config_parser import *
from .activations import LinearActivation, SigmoidActivation, TanhActivation, \
    ReluActivation, IdentityActivation, SoftmaxActivation, BaseActivation
from .evaluators import *
from .poolings import MaxPooling, AvgPooling, MaxWithMaskPooling, BasePoolingType, \
    CudnnAvgPooling, CudnnAvgInclPadPooling, CudnnMaxPooling
from .attrs import *
from .default_decorators import *

try:
    import cPickle as pickle
except ImportError:
    import pickle
import copy

__all__ = [
    'full_matrix_projection',
    'AggregateLevel',
    'ExpandLevel',
    'identity_projection',
    'dotmul_projection',
    'dotmul_operator',
    'repeat_layer',
    'seq_reshape_layer',
    'table_projection',
    'mixed_layer',
    'data_layer',
    'embedding_layer',
    'fc_layer',
    'grumemory',
    'pooling_layer',
    'lstmemory',
    'last_seq',
    'first_seq',
    'cos_sim',
    'l2_distance_layer',
    'hsigmoid',
    'conv_projection',
    'square_error_cost',
    'regression_cost',
    'classification_cost',
    'LayerOutput',
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
    'row_l2_norm_layer',
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
    'BeamInput',
    'cross_entropy_over_beam',
    'multi_binary_label_cross_entropy',
    'sum_cost',
    'rank_cost',
    'lambda_cost',
    'huber_regression_cost',
    'huber_classification_cost',
    'block_expand_layer',
    'maxout_layer',
    'dot_prod_layer',
    'out_prod_layer',
    'printer_layer',
    'print_layer',
    'priorbox_layer',
    'cross_channel_norm_layer',
    'multibox_loss_layer',
    'detection_output_layer',
    'roi_pool_layer',
    'spp_layer',
    'pad_layer',
    'eos_layer',
    'smooth_l1_cost',
    'layer_support',
    'multiplex_layer',
    'row_conv_layer',
    'dropout_layer',
    'prelu_layer',
    'switch_order_layer',
    'gated_unit_layer',
    'crop_layer',
    'sub_nested_seq_layer',
    'clip_layer',
    'slice_projection',
    'seq_slice_layer',
    'kmax_seq_score_layer',
    'img_pool3d_layer',
    'scale_shift_layer',
    'img_conv3d_layer',
    'resize_layer',
    'sub_seq_layer',
    'scale_sub_region_layer',
    'upsample_layer',
    'factorization_machine',
]


class LayerType(object):
    """
    Layer type enumerations.
    """

    DATA = 'data'
    MIXED_LAYER = 'mixed'
    LSTMEMORY = 'lstmemory'
    GRUMEMORY = 'gated_recurrent'
    SEQUENCE_LAST_INSTANCE = 'seqlastins'
    SEQUENCE_FIRST_INSTANCE = 'seqfirstins'
    SEQUENCE_RESHAPE = 'seqreshape'
    POOLING_MAX = 'max'
    POOLING_AVG = 'average'
    UPSAMPLE_LAYER = 'upsample'
    FC_LAYER = 'fc'
    COST = 'cost'
    COSINE_SIM_VEC = 'cos_vm'
    COSINE_SIM = 'cos'
    L2_DISTANCE = 'l2_distance'
    HSIGMOID = 'hsigmoid'
    CONV_LAYER = 'conv'
    CONVTRANS_LAYER = 'convt'
    EXCONV_LAYER = 'exconv'
    EXCONVTRANS_LAYER = 'exconvt'
    CUDNNCONV_LAYER = 'cudnn_conv'
    CUDNNCONVTRANS_LAYER = 'cudnn_convt'
    POOL_LAYER = 'pool'
    POOL3D_LAYER = 'pool3d'
    BATCH_NORM_LAYER = 'batch_norm'
    NORM_LAYER = 'norm'
    SUM_TO_ONE_NORM_LAYER = 'sum_to_one_norm'
    ROW_L2_NORM_LAYER = 'row_l2_norm'
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
    DOT_PROD_LAYER = 'dot_prod'
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
    MULTIPLEX_LAYER = "multiplex"
    ROW_CONV_LAYER = "row_conv"

    PRINT_LAYER = 'print'
    PRIORBOX_LAYER = 'priorbox'
    MULTIBOX_LOSS_LAYER = 'multibox_loss'
    DETECTION_OUTPUT_LAYER = 'detection_output'
    ROI_POOL_LAYER = 'roi_pool'

    CTC_LAYER = 'ctc'
    WARP_CTC_LAYER = 'warp_ctc'
    CRF_LAYER = 'crf'
    CRF_DECODING_LAYER = 'crf_decoding'
    NCE_LAYER = 'nce'

    CONV3D_LAYER = 'conv3d'
    DECONV3D_LAYER = 'deconv3d'

    RANK_COST = 'rank-cost'
    LAMBDA_COST = 'lambda_cost'
    HUBER_REGRESSION = 'huber_regression'
    HUBER_CLASSIFICATION = 'huber_classification'
    CROSS_ENTROPY = 'multi-class-cross-entropy'
    CROSS_ENTROPY_WITH_SELFNORM = 'multi_class_cross_entropy_with_selfnorm'
    CROSS_ENTROPY_OVER_BEAM = 'cross_entropy_over_beam'
    SOFT_BIN_CLASS_CROSS_ENTROPY = 'soft_binary_class_cross_entropy'
    MULTI_BIN_LABEL_CROSS_ENTROPY = 'multi_binary_label_cross_entropy'
    SUM_COST = 'sum_cost'
    SMOOTH_L1 = 'smooth_l1'

    PRELU = 'prelu'
    SWITCH_ORDER_LAYER = 'switch_order'
    CROP_LAYER = 'crop'
    SUB_NESTED_SEQ = 'sub_nested_seq'
    CLIP_LAYER = 'clip'
    SEQ_SLICE = 'seq_slice'

    KMAX_SEQ_SCORE = 'kmax_seq_score'
    SCALE_SHIFT_LAYER = 'scale_shift'

    RESIZE = 'resize'
    SUB_SEQ_LAYER = 'subseq'

    SCALE_SUB_REGION_LAYER = 'scale_sub_region'

    FACTORIZATION_MACHINE = 'factorization_machine'

    @staticmethod
    def is_layer_type(type_name):
        """
        Whether type_name is a layer type.

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
    """
    PaddlePaddle supports three sequence types:

    - :code:`SequenceType.NO_SEQUENCE` means the sample is not a sequence.
    - :code:`SequenceType.SEQUENCE` means the sample is a sequence.
    - :code:`SequenceType.SUB_SEQUENCE` means the sample is a nested sequence,
      each timestep of which is also a sequence.

    Accordingly, AggregateLevel supports two modes:

    - :code:`AggregateLevel.TO_NO_SEQUENCE` means the aggregation acts on each
      timestep of a sequence, both :code:`SUB_SEQUENCE` and :code:`SEQUENCE` will
      be aggregated to :code:`NO_SEQUENCE`.

    - :code:`AggregateLevel.TO_SEQUENCE` means the aggregation acts on each
      sequence of a nested sequence, :code:`SUB_SEQUENCE` will be aggregated to
      :code:`SEQUENCE`.
    """
    TO_NO_SEQUENCE = 'non-seq'
    TO_SEQUENCE = 'seq'
    # compatible with previous configuration
    EACH_TIMESTEP = TO_NO_SEQUENCE
    EACH_SEQUENCE = TO_SEQUENCE


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
    :type parents: list | tuple | collections.Sequence
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
        self.full_name = MakeLayerNameInSubmodel(name)
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

    @property
    def width(self):
        return cp.g_layer_map[self.full_name].width

    @property
    def height(self):
        return cp.g_layer_map[self.full_name].height

    @property
    def depth(self):
        return cp.g_layer_map[self.full_name].depth

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

    2. When used as an independent object like this, you must set the size:

    .. code-block:: python

       proj = full_matrix_projection(input=layer,
                                     size=100,
                                     param_attr=ParamAttr(name='_proj'))

    :param input: The input of this layer.
    :type input: LayerOutput
    :param size: The dimension of this layer.
    :type size: int
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :return: FullMatrixProjection Object.
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
    multiplication, using the transpose of weight.

    ..  math::
        out.row[i] += in.row[i] * w^\mathrm{T}

    :math:`w^\mathrm{T}` means the transpose of weight.
    The simply usage is:

    .. code-block:: python

       proj = trans_full_matrix_projection(input=layer,
                                           size=100,
                                           param_attr=ParamAttr(
                                                name='_proj',
                                                initial_mean=0.0,
                                                initial_std=0.01))

    :param input: The input of this layer.
    :type input: LayerOutput
    :param size: The parameter size. Means the width of parameter.
    :type size: int
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :return: TransposedFullMatrixProjection Object.
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

    2. When used as an independent object like this, you must set the size:

    .. code-block:: python

       proj = table_projection(input=layer,
                               size=100,
                               param_attr=ParamAttr(name='_proj'))


    :param input: The input of this layer, which must contains id fields.
    :type input: LayerOutput
    :param size: The dimension of the output.
    :type size: int
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :return: TableProjection Object.
    :rtype: TableProjection
    """
    proj = TableProjection(
        input_layer_name=input.name, size=size, **param_attr.attr)
    proj.origin = input
    return proj


def identity_projection(input, offset=None, size=None):
    """
    1. If offset=None, it performs IdentityProjection as follows:

    .. math::
       out.row[i] += in.row[i]

    The example usage is:

    .. code-block:: python

       proj = identity_projection(input=layer)


    2. If offset!=None, It executes IdentityOffsetProjection and takes the
       elements of the input in the range [offset, offset+size) as output.

    .. math::
       out.row[i] += in.row[i + \\textrm{offset}]

    The example usage is:

    .. code-block:: python

       proj = identity_projection(input=layer,
                                  offset=10)

    Note that neither of the projections have trainable parameter.

    :param input: The input of this layer.
    :type input: LayerOutput
    :param offset: The offset from the start of the input. The input's
                   elements in the range [offset, offset+size) will be
                   taken as output. If this parameter is not set or set
                   to None, the output will be the same as the input.
    :type offset: int
    :param size: The dimension of this layer. It will be neglected
                 when offset is None or not set.
    :type size: int
    :return: IdentityProjection or IdentityOffsetProjection object
    :rtype: IdentityProjection | IdentityOffsetProjection
    """
    if offset is None:
        proj = IdentityProjection(input_layer_name=input.name)
        proj.origin = input
    else:
        if size is None:
            size = input.size - offset
        proj = IdentityOffsetProjection(
            input_layer_name=input.name, offset=offset, size=size)
        proj.origin = input
    return proj


def slice_projection(input, slices):
    """
    slice_projection slices the input value into multiple parts,
    then selects and merges some of them into a new output.

    .. math::
       output = [input.slices()]

    The example usage is:

    .. code-block:: python

       proj = slice_projection(input=layer, slices=[(0, 10), (20, 30)])

    Note that slice_projection has no trainable parameter.

    :param input: The input of this layer.
    :type input: LayerOutput
    :param slices: A list of start and end offsets of each slice.
    :type slices: list of tuple
    :return: SliceProjection object.
    :rtype: SliceProjection
    """
    assert len(slices) >= 1
    start = 0
    for i in xrange(len(slices)):
        assert len(slices[i]) == 2
        # The start position of the next slice needs to be greater than
        # or equal to the end position of the previous slice.
        assert slices[i][0] >= start
        assert slices[i][1] >= slices[i][0]
        start = slices[i][1]
    proj = SliceProjection(input_layer_name=input.name, slices=slices)
    proj.origin = input
    return proj


@wrap_param_attr_default()
def scaling_projection(input, param_attr=None):
    """
    scaling_projection multiplies the input with a scalar parameter.

    .. math::
       out += w * in

    The example usage is:

    .. code-block:: python

       proj = scaling_projection(input=layer)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :return: ScalingProjection object.
    :rtype: ScalingProjection
    """
    proj = ScalingProjection(input_layer_name=input.name, **param_attr.attr)
    proj.origin = input
    return proj


@wrap_param_attr_default()
def dotmul_projection(input, param_attr=None):
    """
    DotMulProjection takes a layer as input and performs
    element-wise multiplication with weight.

    ..  math::
        out.row[i] += in.row[i] .* weight

    where :math:`.*` means element-wise multiplication.

    The example usage is:

    .. code-block:: python

       proj = dotmul_projection(input=layer)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :return: DotMulProjection object.
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
    scale is a config scalar, its default value is 1.

    The example usage is:

    .. code-block:: python

       op = dotmul_operator(a=layer1, b=layer2, scale=0.5)

    :param a: The first input of this layer.
    :type a: LayerOutput
    :param b: The second input of this layer.
    :type b: LayerOutput
    :param scale: A scalar to scale the product. Its default value is 1.
    :type scale: float
    :return: DotMulOperator object.
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

    It just reorganizes input sequence, combines "context_len" elements of the
    sequence to one context from context_start. "context_start" will be set to
    -(context_len - 1) / 2 by default. When context position is out of sequence
    length, padding will be filled as zero if padding_attr = False, otherwise
    it is trainable.

    For example, origin sequence is [A B C D E F G], context len is 3, padding_attr
    is not set, then after context projection, sequence will
    be [ 0AB ABC BCD CDE DEF EFG FG0 ].

    :param input: The input of this layer, which should be a sequence.
    :type input: LayerOutput
    :param context_len: The length of the context.
    :type context_len: int
    :param context_start: The start position of the context. The default value is
                          -(context_len - 1)/2
    :type context_start: int
    :param padding_attr: Parameter attribute of the padding. If the parameter is
                         set to False, padding will be zero. In other cases, the
                         padding is trainable, and its parameter attribute is set
                         by this parameter.
    :type padding_attr: bool | ParameterAttribute
    :return: Projection object.
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
        :param name: The name of this layer.
        :type name: basestring
        :param size: The dimension of this layer.
        :type size: int
        :param act: Activation type.
        :type act: BaseActivation
        :param bias_attr: The bias attribute. If the parameter is set to False or an object
                          whose type is not ParameterAttribute, no bias is defined. If the
                          parameter is set to True, the bias is initialized to zero.
        :type bias_attr: ParameterAttribute | None | bool | Any
        :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                           details.
        :type layer_attr: ExtraLayerAttribute | None
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
    Mixed Layer. A mixed layer will add all inputs together, then activate the sum.
    Each input is a projection or operator.

    There are two styles of usages.

    1. When the parameter input is not set, use mixed_layer like this:

    .. code-block:: python

       with mixed_layer(size=256) as m:
           m += full_matrix_projection(input=layer1)
           m += identity_projection(input=layer2)

    2. You can also set all inputs when invoke mixed_layer as follows:

    .. code-block:: python

       m = mixed_layer(size=256,
                       input=[full_matrix_projection(input=layer1),
                              full_matrix_projection(input=layer2)])

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param size: The dimension of this layer.
    :type size: int
    :param input: The input of this layer. It is an optional parameter.
    :param act: Activation Type. LinearActivation is the default activation.
    :type act: BaseActivation
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: MixedLayerType object.
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
def data_layer(name, size, depth=None, height=None, width=None,
               layer_attr=None):
    """
    Define DataLayer For NeuralNetwork.

    The example usage is:

    ..  code-block:: python

        data = data_layer(name="input", size=1000)

    :param name: The name of this layer.
    :type name: basestring
    :param size: The dimension of this data layer.
    :type size: int
    :param height: The height of the input image data.
    :type height: int | None
    :param width: The width of the input image data.
    :type width: int | None
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    Layer(
        type=LayerType.DATA,
        name=name,
        size=size,
        depth=depth,
        height=height,
        width=width,
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    if depth is None:
        depth = 1
    num_filters = None
    if height is not None and width is not None:
        num_filters = size / (width * height * depth)
        assert num_filters * width * height * depth == size, \
                "size=%s width=%s height=%s depth=%s" % (size, width, height, depth)

    return LayerOutput(name, LayerType.DATA, size=size, num_filters=num_filters)


@wrap_name_default("embedding")
@wrap_param_attr_default()
@layer_support(ERROR_CLIPPING, DROPOUT)
def embedding_layer(input, size, name=None, param_attr=None, layer_attr=None):
    """
    Define a embedding Layer.

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer, whose type must be Index Data.
    :type input: LayerOutput
    :param size: The dimension of the embedding vector.
    :type size: int
    :param param_attr: The embedding parameter attribute. See ParameterAttribute
                      for details.
    :type param_attr: ParameterAttribute
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute | None
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
    The fully connected layer.

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

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput | list | tuple
    :param size: The dimension of this layer.
    :type size: int
    :param act: Activation Type. TanhActivation is the default activation.
    :type act: BaseActivation
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute | None
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
            if "parameter_name" in param_attr.attr and len(input) > 1:
                logger.fatal(
                    "When the name field of param_attr is manually specified "
                    "and the input is a list, the param_attr should also be a "
                    "list with each item being the param_attr for each input "
                    "item. If only one named param_attr is provided, all the "
                    "input items would share this parameter.")
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
def printer_layer(input, format=None, name=None):
    """
    Print the output value of the layers specified by the parameter input.
    This layer is useful for debugging.

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput | list | tuple
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if isinstance(input, LayerOutput):
        input = [input]
    assert isinstance(input, collections.Sequence)  # list or tuple
    for each in input:
        assert isinstance(each, LayerOutput)

    Layer(
        name=name,
        format=format,
        type=LayerType.PRINT_LAYER,
        inputs=[l.name for l in input], )
    # this layer don't return anything, can not be input of other layer.

# Keep print_layer for compatibility with V1 API.
# 'print_layer' does not work for V2 API because it will be changed to
# 'print' for V2 API. But 'print' is a reserved key word in python.


print_layer = printer_layer


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

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param image: The network input image.
    :type image: LayerOutput
    :param aspect_ratio: The aspect ratio.
    :type aspect_ratio: list
    :param variance: The bounding box variance.
    :type min_size: The minimum size of the priorbox width/height.
    :param min_size: list
    :type max_size: The maximum size of the priorbox width/height. It could be NULL.
    :param max_size: list
    :return: LayerOutput object.
    :rtype: LayerOutput
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


@wrap_name_default("multibox_loss")
def multibox_loss_layer(input_loc,
                        input_conf,
                        priorbox,
                        label,
                        num_classes,
                        overlap_threshold=0.5,
                        neg_pos_ratio=3.0,
                        neg_overlap=0.5,
                        background_id=0,
                        name=None):
    """
    Compute the location loss and the confidence loss for ssd.

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input_loc: The input predicted locations.
    :type input_loc: LayerOutput | List of LayerOutput
    :param input_conf: The input priorbox confidence.
    :type input_conf: LayerOutput | List of LayerOutput
    :param priorbox: The input priorbox location and the variance.
    :type priorbox: LayerOutput
    :param label: The input label.
    :type label: LayerOutput
    :param num_classes: The number of the classification.
    :type num_classes: int
    :param overlap_threshold: The threshold of the overlap.
    :type overlap_threshold: float
    :param neg_pos_ratio: The ratio of the negative bounding box to
                          the positive bounding box.
    :type neg_pos_ratio: float
    :param neg_overlap: The negative bounding box overlap threshold.
    :type neg_overlap: float
    :param background_id: The background class index.
    :type background_id: int
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if isinstance(input_loc, LayerOutput):
        input_loc = [input_loc]
    assert isinstance(input_loc, collections.Sequence)  # list or tuple
    for each in input_loc:
        assert isinstance(each, LayerOutput)
    input_loc_num = len(input_loc)

    if isinstance(input_conf, LayerOutput):
        input_conf = [input_conf]
    assert isinstance(input_conf, collections.Sequence)  # list or tuple
    for each in input_conf:
        assert isinstance(each, LayerOutput)
    input_conf_num = len(input_conf)
    # Check the input layer number.
    assert input_loc_num == input_conf_num

    inputs = [priorbox.name, label.name]
    inputs.extend([l.name for l in input_loc])
    inputs.extend([l.name for l in input_conf])
    parents = [priorbox, label]
    parents.extend(input_loc)
    parents.extend(input_conf)

    Layer(
        name=name,
        type=LayerType.MULTIBOX_LOSS_LAYER,
        inputs=inputs,
        input_num=input_loc_num,
        num_classes=num_classes,
        overlap_threshold=overlap_threshold,
        neg_pos_ratio=neg_pos_ratio,
        neg_overlap=neg_overlap,
        background_id=background_id)
    return LayerOutput(
        name, LayerType.MULTIBOX_LOSS_LAYER, parents=parents, size=1)


@wrap_name_default("detection_output")
def detection_output_layer(input_loc,
                           input_conf,
                           priorbox,
                           num_classes,
                           nms_threshold=0.45,
                           nms_top_k=400,
                           keep_top_k=200,
                           confidence_threshold=0.01,
                           background_id=0,
                           name=None):
    """
    Apply the NMS to the output of network and compute the predict bounding
    box location. The output's shape of this layer could be zero if there is
    no valid bounding box.

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input_loc: The input predict locations.
    :type input_loc: LayerOutput | List of LayerOutput.
    :param input_conf: The input priorbox confidence.
    :type input_conf: LayerOutput | List of LayerOutput.
    :param priorbox: The input priorbox location and the variance.
    :type priorbox: LayerOutput
    :param num_classes: The number of the classes.
    :type num_classes: int
    :param nms_threshold: The Non-maximum suppression threshold.
    :type nms_threshold: float
    :param nms_top_k: The bounding boxes number kept of the NMS's output.
    :type nms_top_k: int
    :param keep_top_k: The bounding boxes number kept of the layer's output.
    :type keep_top_k: int
    :param confidence_threshold: The classification confidence threshold.
    :type confidence_threshold: float
    :param background_id: The background class index.
    :type background_id: int
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if isinstance(input_loc, LayerOutput):
        input_loc = [input_loc]
    assert isinstance(input_loc, collections.Sequence)  # list or tuple
    for each in input_loc:
        assert isinstance(each, LayerOutput)
    input_loc_num = len(input_loc)

    if isinstance(input_conf, LayerOutput):
        input_conf = [input_conf]
    assert isinstance(input_conf, collections.Sequence)  # list or tuple
    for each in input_conf:
        assert isinstance(each, LayerOutput)
    input_conf_num = len(input_conf)

    # Check the input layer number.
    assert input_loc_num == input_conf_num

    inputs = [priorbox.name]
    inputs.extend([l.name for l in input_loc])
    inputs.extend([l.name for l in input_conf])
    parents = [priorbox]
    parents.extend(input_loc)
    parents.extend(input_conf)

    size = keep_top_k * 7

    Layer(
        name=name,
        type=LayerType.DETECTION_OUTPUT_LAYER,
        inputs=inputs,
        size=size,
        input_num=input_loc_num,
        num_classes=num_classes,
        nms_threshold=nms_threshold,
        nms_top_k=nms_top_k,
        keep_top_k=keep_top_k,
        confidence_threshold=confidence_threshold,
        background_id=background_id)
    return LayerOutput(
        name, LayerType.DETECTION_OUTPUT_LAYER, parents=parents, size=size)


@wrap_name_default("roi_pool")
def roi_pool_layer(input,
                   rois,
                   pooled_width,
                   pooled_height,
                   spatial_scale,
                   num_channels=None,
                   name=None):
    """
    A layer used by Fast R-CNN to extract feature maps of ROIs from the last
    feature map.

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input layer.
    :type input: LayerOutput.
    :param rois: The input ROIs' data.
    :type rois: LayerOutput.
    :param pooled_width: The width after pooling.
    :type pooled_width: int
    :param pooled_height: The height after pooling.
    :type pooled_height: int
    :param spatial_scale: The spatial scale between the image and feature map.
    :type spatial_scale: float
    :param num_channels: The number of the input channels.
    :type num_channels: int
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if num_channels is None:
        assert input.num_filters is not None
        num_channels = input.num_filters
    size = num_channels * pooled_width * pooled_height
    Layer(
        name=name,
        type=LayerType.ROI_POOL_LAYER,
        inputs=[input.name, rois.name],
        pooled_width=pooled_width,
        pooled_height=pooled_height,
        spatial_scale=spatial_scale,
        num_channels=num_channels)
    return LayerOutput(
        name, LayerType.ROI_POOL_LAYER, parents=[input, rois], size=size)


@wrap_name_default("cross_channel_norm")
def cross_channel_norm_layer(input, name=None, param_attr=None):
    """
    Normalize a layer's output. This layer is necessary for ssd. This
    layer applys normalization across the channels of each sample to
    a convolutional layer's output and scales the output by a group of
    trainable factors whose dimensions equal to the channel's number.

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
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
                  agg_level=AggregateLevel.TO_NO_SEQUENCE,
                  stride=-1,
                  layer_attr=None):
    """
    Pooling layer for sequence inputs, not used for Image.

    If stride > 0, this layer slides a window whose size is determined by stride,
    and returns the pooling value of the sequence in the window as the output. Thus,
    a long sequence will be shortened. Note that for sequence with sub-sequence, the
    default value of stride is -1.

    The example usage is:

    .. code-block:: python

       seq_pool = pooling_layer(input=layer,
                                pooling_type=AvgPooling(),
                                agg_level=AggregateLevel.TO_NO_SEQUENCE)

    :param agg_level: AggregateLevel.TO_NO_SEQUENCE or
                      AggregateLevel.TO_SEQUENCE
    :type agg_level: AggregateLevel
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param pooling_type: Type of pooling. MaxPooling is the default pooling.
    :type pooling_type: BasePoolingType | None
    :param stride: The step size between successive pooling regions.
    :type stride: int
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute | None
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

    if agg_level == AggregateLevel.TO_SEQUENCE:
        assert stride == -1

    Layer(
        name=name,
        type=pooling_type.name,
        inputs=[Input(input.name)],
        bias=ParamAttr.to_bias(bias_attr),
        trans_type=agg_level,
        stride=stride,
        **extra_dict)

    return LayerOutput(
        name, pooling_type.name, parents=[input], size=input.size)


@wrap_bias_attr_default()
@wrap_param_attr_default()
@wrap_act_default(param_names=['gate_act'], act=SigmoidActivation())
@wrap_act_default(param_names=["act", 'state_act'], act=TanhActivation())
@wrap_name_default("lstmemory")
@layer_support()
def lstmemory(input,
              name=None,
              size=None,
              reverse=False,
              act=None,
              gate_act=None,
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

    Reference:
        `Generating Sequences With Recurrent Neural Networks
        <https://arxiv.org/pdf/1308.0850.pdf>`_

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param size: DEPRECATED. The dimension of the lstm cell.
    :type size: int
    :param input: The input of this layer.
    :type input: LayerOutput
    :param reverse: Whether the input sequence is processed in a reverse order.
    :type reverse: bool
    :param act: Activation type. TanhActivation is the default activation.
    :type act: BaseActivation
    :param gate_act: Activation type of this layer's gates. SigmoidActivation is the
                     default activation.
    :type gate_act: BaseActivation
    :param state_act: Activation type of the state. TanhActivation is the default activation.
    :type state_act: BaseActivation
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute | None
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
        plog("size of lstmemory layer: %s is automatically set to "
             "size of input layer / 4. The parameter size passing to "
             "this layer is ignored." % (name))

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
@layer_support()
def grumemory(input,
              size=None,
              name=None,
              reverse=False,
              act=None,
              gate_act=None,
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
    :math:`W_{r}x_{t}`, :math:`W_{z}x_{t}` and :math:`W x_t` are not performed
    in gate_recurrent layer. Consequently, an additional mixed_layer with
    full_matrix_projection or a fc_layer must be included before grumemory
    is called.

    Reference:
        `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
        <https://arxiv.org/abs/1412.3555>`_

    The simple usage is:

    .. code-block:: python

       gru = grumemory(input)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput.
    :param size: DEPRECATED. The dimension of the gru cell.
    :type size: int
    :param reverse: Whether the input sequence is processed in a reverse order.
    :type reverse: bool
    :param act: Activation type, TanhActivation is the default. This activation
                affects the :math:`{\\tilde{h_t}}`.
    :type act: BaseActivation
    :param gate_act: Activation type of this layer's two gates. SigmoidActivation is
                     the default activation. This activation affects the :math:`z_t`
                     and :math:`r_t`. It is the :math:`\\sigma` in the above formula.
    :type gate_act: BaseActivation
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute | None
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
        plog("size of grumemory layer: %s is automatically set to "
             "size of input layer / 3. The parameter size passing to this "
             "layer is ignored." % (name))

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
             agg_level=AggregateLevel.TO_NO_SEQUENCE,
             stride=-1,
             layer_attr=None):
    """
    Get Last Timestamp Activation of a sequence.

    If stride > 0, this layer will slide a window whose size is determined by stride,
    and return the last value of the sequence in the window as the output. Thus, a
    long sequence will be shortened. Note that for sequence with sub-sequence, the
    default value of stride is -1.

    The simple usage is:

    .. code-block:: python

       seq = last_seq(input=layer)

    :param agg_level: Aggregated level
    :type agg_level: AggregateLevel
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param stride: The step size between successive pooling regions.
    :type stride: int
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if input.reverse is not None and input.reverse:
        logger.warning("You are getting the last instance of a sequence that"
                       " is a output of a REVERSED layer. There is no time"
                       " series information at all. Maybe you want to use"
                       " first_seq instead.")

    if agg_level == AggregateLevel.TO_SEQUENCE:
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
              agg_level=AggregateLevel.TO_NO_SEQUENCE,
              stride=-1,
              layer_attr=None):
    """
    Get First Timestamp Activation of a sequence.

    If stride > 0, this layer will slide a window whose size is determined by stride,
    and return the first value of the sequence in the window as the output. Thus, a
    long sequence will be shortened. Note that for sequence with sub-sequence, the
    default value of stride is -1.

    The simple usage is:

    .. code-block:: python

       seq = first_seq(input=layer)

    :param agg_level: aggregation level
    :type agg_level: AggregateLevel
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param stride: The step size between successive pooling regions.
    :type stride: int
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    if input.reverse is not None and not input.reverse:
        logger.warning('You are getting the first instance for a time series,'
                       ' and it is a normal recurrent layer output. There is no'
                       ' time series information at all. Maybe you want to use'
                       ' last_seq instead.')

    if agg_level == AggregateLevel.TO_SEQUENCE:
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
    """
    Please refer to AggregateLevel first.

    ExpandLevel supports two modes:

    - :code:`ExpandLevel.FROM_NO_SEQUENCE` means the expansion acts on
      :code:`NO_SEQUENCE`, which will be expanded to
      :code:`SEQUENCE` or :code:`SUB_SEQUENCE`.

    - :code:`ExpandLevel.FROM_SEQUENCE` means the expansion acts on
      :code:`SEQUENCE`, which will be expanded to
      :code:`SUB_SEQUENCE`.
    """
    FROM_NO_SEQUENCE = AggregateLevel.TO_NO_SEQUENCE
    FROM_SEQUENCE = AggregateLevel.TO_SEQUENCE
    # compatible with previous configuration
    FROM_TIMESTEP = FROM_NO_SEQUENCE


@wrap_name_default()
@layer_support()
def expand_layer(input,
                 expand_as,
                 name=None,
                 bias_attr=False,
                 expand_level=ExpandLevel.FROM_NO_SEQUENCE,
                 layer_attr=None):
    """
    A layer for expanding dense data or (sequence data where the length of each
    sequence is one) to sequence data.

    The example usage is:

    .. code-block:: python

       expand = expand_layer(input=layer1,
                             expand_as=layer2,
                             expand_level=ExpandLevel.FROM_NO_SEQUENCE)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param expand_as: Expand the input according to this layer's sequence infomation. And
                      after the operation, the input expanded will have the same number of
                      elememts as this layer.
    :type expand_as: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param expand_level: Whether the input layer is a sequence or the element of a sequence.
    :type expand_level: ExpandLevel
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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
@wrap_act_default(act=IdentityActivation())
@layer_support()
def repeat_layer(input,
                 num_repeats,
                 as_row_vector=True,
                 act=None,
                 name=None,
                 layer_attr=None):
    """
    A layer for repeating the input for num_repeats times.

    If as_row_vector:

    .. math::
       y  = [x_1,\cdots, x_n, \cdots, x_1, \cdots, x_n]

    If not as_row_vector:

    .. math::
       y  = [x_1,\cdots, x_1, \cdots, x_n, \cdots, x_n]


    The example usage is:

    .. code-block:: python

       expand = repeat_layer(input=layer, num_repeats=4)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param num_repeats: The times of repeating the input.
    :type num_repeats: int
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param as_row_vector: Whether to treat the input as row vectors or not. If
                          the parameter is set to True, the repeating operation
                          will be performed in the column direction. Otherwise,
                          it will be performed in the row direction.
    :type as_row_vector: bool
    :param act: Activation type. IdentityActivation is the default activation.
    :type act: BaseActivation
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    l = Layer(
        inputs=[input.name],
        name=name,
        active_type=act.name,
        num_filters=num_repeats,
        as_row_vector=as_row_vector,
        type=LayerType.FEATURE_MAP_EXPAND_LAYER,
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        size=l.config.size,
        layer_type=LayerType.FEATURE_MAP_EXPAND_LAYER,
        activation=act,
        parents=[input])


@wrap_name_default("seqreshape")
@wrap_act_default(act=IdentityActivation())
@wrap_bias_attr_default(has_bias=False)
@layer_support(ERROR_CLIPPING, DROPOUT)
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

    :param input: The input of this layer.
    :type input: LayerOutput
    :param reshape_size: The dimension of the reshaped sequence.
    :type reshape_size: int
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param act: Activation type. IdentityActivation is the default activation.
    :type act: BaseActivation
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute.
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
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
    This layer performs linear interpolation on two inputs,
    which is used in NEURAL TURING MACHINE.

    .. math::
       y.row[i] = w[i] * x_1.row[i] + (1 - w[i]) * x_2.row[i]

    where :math:`x_1` and :math:`x_2` are two (batchSize x dataDim) inputs,
    :math:`w` is (batchSize x 1) weight vector, and :math:`y` is
    (batchSize x dataDim) output.

    The example usage is:

    .. code-block:: python

       interpolation = interpolation_layer(input=[layer1, layer2], weight=layer3)

    :param input: The input of this layer.
    :type input: list | tuple
    :param weight: Weight layer.
    :type weight: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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
    This layer implements bilinear interpolation on convolutional layer's output.

    Please refer to Wikipedia: https://en.wikipedia.org/wiki/Bilinear_interpolation

    The simple usage is:

    .. code-block:: python

       bilinear = bilinear_interp_layer(input=layer1, out_size_x=64, out_size_y=64)

    :param input: The input of this layer.
    :type input: LayerOutput.
    :param out_size_x: The width of the output.
    :type out_size_x: int
    :param out_size_y: The height of the output.
    :type out_size_y: int
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
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

    where :math:`x` is an input vector, :math:`w` is a scalar exponent,
    and :math:`y` is an output vector.

    The example usage is:

    .. code-block:: python

       power = power_layer(input=layer1, weight=layer2)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param weight: The exponent of the power.
    :type weight: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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

    :param input: The input of this layer.
    :type input: LayerOutput
    :param weight: The weight of each sample.
    :type weight: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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

    :param input: The input of this layer.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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

    :param input: The input of this layer.
    :type input: LayerOutput
    :param height: The height of the sample matrix.
    :type height: int
    :param width: The width of the sample matrix.
    :type width: int
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param a: The first input of this layer.
    :type a: LayerOutput
    :param b: The second input of this layer.
    :type b: LayerOutput
    :param scale: The scale of the cosine similarity. 1 is the default value.
    :type scale: float
    :param size: The dimension of this layer. NOTE size_a * size should equal size_b.
    :type size: int
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for details.
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
@layer_support()
def l2_distance_layer(x, y, name=None, layer_attr=None):
    """
    This layer calculates and returns the Euclidean distance between two input
    vectors x and y. The equation is as follows:

    ..  math::
        l2_distance(\\mathbf{x}, \\mathbf{y}) = \\sqrt{\\sum_{i=1}^D(x_i - y_i)}

    The output size of this layer is fixed to be 1. Note that the above
    computation is for one sample. Multiple samples are processed in one batch.

    The example usage is:

    .. code-block:: python

       l2_sim = l2_distance(x=layer1, y=layer2)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param x: The first input x for this layer, whose output is a matrix with
              dimensionality N x D. N is the sample number in a mini-batch.
              D is the dimensionality of x's output.
    :type x: LayerOutput
    :param y: The second input y for this layer, whose output is a matrix with
              dimensionality N x D. N is the sample number in a mini-batch.
              D is the dimensionality of y's output.
    :type y: LayerOutput
    :param layer_attr: The extra layer attributes, for example, drop rate.
                       See ExtraLayerAttribute for more details.
    :type layer_attr: ExtraLayerAttribute
    :return: The returned LayerOutput object.
    :rtype: LayerOutput
    """

    assert isinstance(x, LayerOutput) and isinstance(y, LayerOutput)
    Layer(
        name=name,
        type=LayerType.L2_DISTANCE,
        inputs=[x.name, y.name],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(name, LayerType.L2_DISTANCE, parents=[x, y], size=1)


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

    Reference:
        `Hierarchical Probabilistic Neural Network Language Model
        <http://www.gatsby.ucl.ac.uk/aistats/fullpapers/208.pdf>`_

    The example usage is:

    ..  code-block:: python

        cost = hsigmoid(input=[layer1, layer2],
                        label=data_layer)

    :param input: The input of this layer.
    :type input: LayerOutput | list | tuple
    :param label: The input label.
    :type label: LayerOutput
    :param num_classes: The number of classes. And it should be larger than 2. If the parameter
                        is not set or set to None, its actual value will be automatically set to
                        the number of labels.
    :type num_classes: int
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for details.
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
                   dilation=1,
                   bias_attr=None,
                   param_attr=None,
                   shared_biases=True,
                   layer_attr=None,
                   filter_size_y=None,
                   stride_y=None,
                   padding_y=None,
                   dilation_y=None,
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
    num_filters.

    There are several groups of filters in PaddlePaddle implementation.
    If the groups attribute is greater than 1, for example groups=2,
    the input will be splitted into 2 parts along the channel axis, and
    the filters will also be splitted into 2 parts. The first half of the filters 
    is only connected to the first half of the input channels, while the second 
    half of the filters is only connected to the second half of the input. After
    the computation of convolution for each part of input,
    the output will be obtained by concatenating the two results.

    The details of grouped convolution, please refer to:
    `ImageNet Classification With Deep Convolutional Neural Networks
    <http://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf>`_
    
    The example usage is:

    ..  code-block:: python

        conv = img_conv_layer(input=data, filter_size=1, filter_size_y=1,
                              num_channels=8,
                              num_filters=16, stride=1,
                              bias_attr=False,
                              act=ReluActivation())

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param filter_size: The dimensions of the filter kernel. If the parameter is
                        set to one integer, the two dimensions on x and y axises
                        will be same when filter_size_y is not set. If it is set
                        to a list, the first element indicates the dimension on
                        the x axis, and the second is used to specify the dimension
                        on the y axis when filter_size_y is not provided.
    :type filter_size: int | tuple | list
    :param filter_size_y: The dimension of the filter kernel on the y axis. If the parameter
                          is not set, it will be set automatically according to filter_size.
    :type filter_size_y: int
    :param num_filters: The number of filters. It is as same as the output image channel.
    :type num_filters: int
    :param act: Activation type. ReluActivation is the default activation.
    :type act: BaseActivation
    :param groups: The group number. 1 is the default group number.
    :type groups: int
    :param stride: The strides. If the parameter is set to one integer, the strides
                   on x and y axises will be same when stride_y is not set. If it is
                   set to a list, the first element indicates the stride on the x axis,
                   and the second is used to specify the stride on the y axis when
                   stride_y is not provided. 1 is the default value.
    :type stride: int | tuple | list
    :param stride_y: The stride on the y axis.
    :type stride_y: int
    :param padding: The padding sizes. If the parameter is set to one integer, the padding
                    sizes on x and y axises will be same when padding_y is not set. If it
                    is set to a list, the first element indicates the padding size on the
                    x axis, and the second is used to specify the padding size on the y axis
                    when padding_y is not provided. 0 is the default padding size.
    :type padding: int | tuple | list
    :param padding_y: The padding size on the y axis.
    :type padding_y: int
    :param dilation: The dimensions of the dilation. If the parameter is set to one integer,
                     the two dimensions on x and y axises will be same when dilation_y is not
                     set. If it is set to a list, the first element indicates the dimension
                     on the x axis, and the second is used to specify the dimension on the y
                     axis when dilation_y is not provided. 1 is the default dimension.
    :type dilation: int | tuple | list
    :param dilation_y: The dimension of the dilation on the y axis.
    :type dilation_y: int
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param num_channels: The number of input channels. If the parameter is not set or
                         set to None, its actual value will be automatically set to
                         the channel number of the input.
    :type num_channels: int
    :param param_attr: The parameter attribute. See ParameterAttribute for
                       details.
    :type param_attr: ParameterAttribute
    :param shared_biases: Whether biases will be shared between filters or not.
    :type shared_biases: bool
    :param layer_attr: The extra layer attributes. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :param trans: True if it is a convTransLayer, False if it is a convLayer
    :type trans: bool
    :param layer_type: Specify the layer type. If the dilation's dimension on one axis is
                       larger than 1, layer_type has to be "cudnn_conv" or "cudnn_convt".
                       If trans=True, layer_type has to be "exconvt" or "cudnn_convt",
                       otherwise layer_type has to be either "exconv" or "cudnn_conv".
    :type layer_type: basestring
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

    if dilation_y is None:
        if isinstance(dilation, collections.Sequence):
            assert len(dilation) == 2
            dilation, dilation_y = dilation
        else:
            dilation_y = dilation

    if param_attr.attr.get('initial_smart'):
        # special initial for conv layers.
        init_w = (2.0 / (filter_size**2 * num_channels))**0.5
        param_attr.attr["initial_mean"] = 0.0
        param_attr.attr["initial_std"] = init_w
        param_attr.attr["initial_strategy"] = 0
        param_attr.attr["initial_smart"] = False

    if layer_type:
        if dilation > 1 or dilation_y > 1:
            assert layer_type in [
                "cudnn_conv", "cudnn_convt", "exconv", "exconvt"
            ]
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
                dilation=dilation,
                stride=stride,
                channels=num_channels,
                groups=groups,
                filter_size_y=filter_size_y,
                padding_y=padding_y,
                dilation_y=dilation_y,
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
                   ceil_mode=True,
                   exclude_mode=None):
    """
    Image pooling Layer.

    The details of pooling layer, please refer to ufldl's pooling_ .

    .. _pooling: http://ufldl.stanford.edu/tutorial/supervised/Pooling/

    - ceil_mode=True:

    ..  math::

        w & = 1 + ceil(\\frac{input\_width + 2 * padding - pool\_size}{stride})

        h & = 1 + ceil(\\frac{input\_height + 2 * padding\_y - pool\_size\_y}{stride\_y})

    - ceil_mode=False:

    ..  math::

        w & = 1 + floor(\\frac{input\_width + 2 * padding - pool\_size}{stride})

        h & = 1 + floor(\\frac{input\_height + 2 * padding\_y - pool\_size\_y}{stride\_y})

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

    :param padding: The padding size on the x axis. 0 is the default padding size.
    :type padding: int
    :param padding_y: The padding size on the y axis. If the parameter is not set
                      or set to None, it will be set to 'padding' automatically.
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param pool_size: The pooling window length on the x axis.
    :type pool_size: int
    :param pool_size_y: The pooling window length on the y axis. If the parameter is
                        not set or set to None, its actual value will be automatically
                        set to pool_size.
    :type pool_size_y: int
    :param num_channels: The number of input channels. If the parameter is not set or
                         set to None, its actual value will be automatically set to
                         the channels number of the input.
    :type num_channels: int
    :param pool_type: Pooling type. MaxPooling is the default pooling.
    :type pool_type: BasePoolingType
    :param stride: The stride on the x axis. 1 is the default value.
    :type stride: int
    :param stride_y: The stride on the y axis. If the parameter is not set or set to
                     None, its actual value will be automatically set to 'stride'.
    :type stride_y: int
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :param ceil_mode: Whether to use the ceil function to calculate output height and width.
                      True is the default. If it is set to False, the floor function will
                      be used.
    :type ceil_mode: bool
    :param exclude_mode: Whether to exclude the padding cells when calculating, but only 
                         work when pool_type is AvgPooling. If None, also exclude the padding 
                         cells. If use cudnn, use CudnnAvgPooling or CudnnAvgInclPadPooling 
                         as pool_type to identify the mode.
    :type exclude_mode: bool
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

    assert type(pool_type) in [AvgPooling, MaxPooling, MaxWithMaskPooling, CudnnAvgPooling,
                               CudnnMaxPooling, CudnnAvgInclPadPooling], \
        "only (Cudnn)AvgPooling, (Cudnn)MaxPooling, MaxWithMaskPooling are supported"

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
        exclude_mode=exclude_mode,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        LayerType.POOL_LAYER,
        parents=[input],
        num_filters=num_channels,
        size=l.config.size)


@wrap_name_default("pool3d")
@layer_support()
def img_pool3d_layer(input,
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
                     pool_size_z=None,
                     stride_z=None,
                     padding_z=None,
                     ceil_mode=True):
    """
    Image pooling Layer.

    The details of pooling layer, please refer ufldl's pooling_ .

    .. _pooling: http://ufldl.stanford.edu/tutorial/supervised/Pooling/

    - ceil_mode=True:

    ..  math::

        w & = 1 + \\frac{ceil(input\_width + 2 * padding - pool\_size)}{stride}

        h & = 1 + \\frac{ceil(input\_height + 2 * padding\_y - pool\_size\_y)}{stride\_y}

        d & = 1 + \\frac{ceil(input\_depth + 2 * padding\_z - pool\_size\_z)}{stride\_z}

    - ceil_mode=False:

    ..  math::

        w & = 1 + \\frac{floor(input\_width + 2 * padding - pool\_size)}{stride}

        h & = 1 + \\frac{floor(input\_height + 2 * padding\_y - pool\_size\_y)}{stride\_y}

        d & = 1 + \\frac{floor(input\_depth + 2 * padding\_z - pool\_size\_z)}{stride\_z}

    The example usage is:

    ..  code-block:: python

        maxpool = img_pool3d_layer(input=conv,
                                 pool_size=3,
                                 num_channels=8,
                                 stride=1,
                                 padding=1,
                                 pool_type=MaxPooling())

    :param padding: pooling padding width.
    :type padding: int | tuple | list
    :param name: The name of this layer. It is optional.
    :type name: basestring.
    :param input: The input of this layer.
    :type input: LayerOutput
    :param pool_size: The pooling window lengths along three axises. If the parameter
                      is set to one integer, the three lengths will be same.
    :type pool_size: int | tuple | list
    :param num_channels: The number of input channels. If the parameter is not set or
                         set to None, its actual value will be automatically set to
                         the channels number of the input.
    :type num_channels: int
    :param pool_type: Pooling type. MaxPooling is the default pooling.
    :type pool_type: BasePoolingType
    :param stride: The strides of the pooling along three axises. If the parameter
                   is set to one integer, the three strides will be same. 1 is the
                   default value.
    :type stride: int | tuple | list
    :param padding: The sizes of padding along three axises. If the parameter is set to
                    one integer, they will be same. 0 is the default padding size.
    :type padding: int | tuple | list
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :param ceil_mode: Wether to use the ceil function to calculate output height and width.
                      True is the default. If it is set to False, the floor function will
                      be used.
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

    if isinstance(pool_size, collections.Sequence):
        assert len(pool_size) == 3
        pool_size, pool_size_y, pool_size_z = pool_size
    else:
        pool_size_y = pool_size
        pool_size_z = pool_size

    if isinstance(stride, collections.Sequence):
        assert len(stride) == 3
        stride, stride_y, stride_z = stride
    else:
        stride_y = stride
        stride_z = stride

    if isinstance(padding, collections.Sequence):
        assert len(padding) == 3
        padding, padding_y, padding_y = padding
    else:
        padding_y = padding
        padding_z = padding

    l = Layer(
        name=name,
        type=LayerType.POOL3D_LAYER,
        inputs=[
            Input(
                input.name,
                pool=Pool3d(
                    pool_type=type_name,
                    channels=num_channels,
                    size_x=pool_size,
                    start=None,
                    stride=stride,
                    padding=padding,
                    size_y=pool_size_y,
                    stride_y=stride_y,
                    padding_y=padding_y,
                    size_z=pool_size_z,
                    stride_z=stride_z,
                    padding_z=padding_z))
        ],
        ceil_mode=ceil_mode,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        LayerType.POOL_LAYER,
        parents=[input],
        num_filters=num_channels,
        size=l.config.size)


@wrap_name_default("upsample")
@layer_support()
def upsample_layer(input,
                   name=None,
                   scale=None,
                   scale_y=None,
                   upsample_size=None,
                   upsample_size_y=None,
                   pad_out_x=False,
                   pad_out_y=False,
                   layer_attr=None):
    """
    The DePooling process.
    Inputs should be a list of length 2. The first input is a layer,
    and the second input should be the MaxWithMaskPoolingLayer

    The example usage is:

    ..  code-block:: python
        pool1 = paddle.v2.layer.img_pool(input=input, pool_size=2, stride=2,
                                        pool_type=paddle.pooling.MaxWithMask())
        upsample = paddle.v2.layer.upsample(input=[layer1, pool1])

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: contains an input layer and a MaxWithMaskPoolingLayer
    :type input: list | tuple | collections.Sequence
    :param scale: outputSize =  scale * inputSize
    :type scale: int | list | tuple | .
    :param scale_y: scale_y will be equal to scale, if it's value is None, 
    :type scale: int | None. 
    :param upsample_size: specify the outputSize.
    :type upsample_size: int | list | tuple.
    :param upsample_size_y: specify the y dimension outputSize.
    :type upsample_size_y: int.
    :param pad_out_x: specify exact x dimension size. This parameter only works when scale is 2
    :type pad_out_x: bool.
    :param pad_out_y: specify exact y dimension size. This parameter only works when scale is 2
    :type pad_out_y: bool.
    :param layer_attr: Extra Layer Attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert (scale is not None) or (upsample_size is not None), \
            'scale or upsample_size, there must be one to be designated'

    assert len(input) == 2, 'layer input size must be 2'

    assert input[1].layer_type == LayerType.POOL_LAYER, \
            'the second input should be the MaxPoolWithMaskLayer'

    scale_y = scale \
            if scale is not None else scale_y
    upsample_size_y = upsample_size  \
            if upsample_size is not None else upsample_size_y

    layer_type = LayerType.UPSAMPLE_LAYER

    layer = Layer(
        name=name,
        type=layer_type,
        inputs=[
            Input(
                input[0].name,
                upsample=Upsample(scale, scale_y, pad_out_x, pad_out_y,
                                  upsample_size, upsample_size_y)),
            Input(input[1].name)
        ],
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    sz = layer.config.size

    return LayerOutput(name, layer_type=layer_type, parents=input, size=sz)


@wrap_name_default("spp")
@layer_support()
def spp_layer(input,
              name=None,
              num_channels=None,
              pool_type=None,
              pyramid_height=None,
              layer_attr=None):
    """
    A layer performs spatial pyramid pooling.

    Reference:
        `Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
        <https://arxiv.org/abs/1406.4729>`_

    The example usage is:

    ..  code-block:: python

        spp = spp_layer(input=data,
                        pyramid_height=2,
                        num_channels=16,
                        pool_type=MaxPooling())

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param num_channels: The number of input channels. If the parameter is not set or
                         set to None, its actual value will be automatically set to
                         the channels number of the input.
    :type num_channels: int
    :param pool_type: Pooling type. MaxPooling is the default pooling.
    :type scale: BasePoolingType
    :param pyramid_height: The pyramid height of this pooling.
    :type pyramid_height: int
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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

    Reference:
        `ImageNet Classification with Deep Convolutional Neural Networks
        <http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf>`_

    The example usage is:

    ..  code-block:: python

        norm = img_cmrnorm_layer(input=net, size=5)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param size: Normalize in number of :math:`size` feature maps.
    :type size: int
    :param scale: The hyper-parameter.
    :type scale: float
    :param power: The hyper-parameter.
    :type power: float
    :param num_channels: The number of input channels. If the parameter is not set or
                         set to None, its actual value will be automatically set to
                         the channels number of the input.
    :param layer_attr: The extra layer attributes. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    return __img_norm_layer__(name, input, size, "cmrnorm-projection", scale,
                              power, num_channels, 0, layer_attr)


@wrap_bias_attr_default()
@wrap_param_attr_default(
    default_factory=lambda _: ParamAttr(initial_mean=1.0, initial_std=0.))
@wrap_act_default(act=ReluActivation())
@wrap_name_default("batch_norm")
@layer_support(DROPOUT, ERROR_CLIPPING)
def batch_norm_layer(input,
                     act=None,
                     name=None,
                     img3D=False,
                     num_channels=None,
                     bias_attr=None,
                     param_attr=None,
                     layer_attr=None,
                     batch_norm_type=None,
                     epsilon=1e-5,
                     moving_average_fraction=0.9,
                     use_global_stats=None,
                     mean_var_names=None):
    """
    Batch Normalization Layer. The notation of this layer is as follows.

    :math:`x` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    Reference:
        `Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift
        <http://arxiv.org/abs/1502.03167>`_

    The example usage is:

    ..  code-block:: python

        norm = batch_norm_layer(input=net, act=ReluActivation())

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: This layer's input which is to be performed batch normalization on.
    :type input: LayerOutput
    :param batch_norm_type: We have batch_norm, mkldnn_batch_norm and cudnn_batch_norm.
                            batch_norm supports CPU, MKLDNN and GPU. cudnn_batch_norm
                            requires cuDNN version greater or equal to v4 (>=v4).
                            But cudnn_batch_norm is faster and needs less
                            memory than batch_norm. mkldnn_batch_norm requires
                            use_mkldnn is enabled. By default (None), we will
                            automatically select cudnn_batch_norm for GPU,
                            mkldnn_batch_norm for MKLDNN and batch_norm for CPU.
                            Users can specify the batch norm type. If you use
                            cudnn_batch_norm, we suggested you use latest version,
                            such as v5.1.
    :type batch_norm_type: None | string, None or "batch_norm" or "cudnn_batch_norm"
                           or "mkldnn_batch_norm"
    :param act: Activation type. ReluActivation is the default activation.
    :type act: BaseActivation
    :param num_channels: The number of input channels. If the parameter is not set or
                         set to None, its actual value will be automatically set to
                         the channels number of the input.
    :type num_channels: int
    :param bias_attr: :math:`\\beta`. The bias attribute. If the parameter is set to
                      False or an object whose type is not ParameterAttribute, no
                      bias is defined. If the parameter is set to True, the bias is
                      initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param param_attr: :math:`\\gamma`. The parameter attribute. See ParameterAttribute
                       for details.
    :type param_attr: ParameterAttribute
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :param use_global_stats: Whether use moving mean/variance statistics during
                             testing peroid. If the parameter is set to None or
                             True, it will use moving mean/variance statistics
                             during testing. If the parameter is set to False, it
                             will use the mean and variance of the current batch
                             of test data.
    :type use_global_stats: bool | None.
    :param epsilon: The small constant added to the variance to improve numeric stability.
    :type epsilon: float.
    :param moving_average_fraction: Factor used in the moving average computation.
                                   :math:`runningMean = newMean*(1-factor) + runningMean*factor`
    :type moving_average_fraction: float.
    :param mean_var_names: [mean name, variance name]
    :type mean_var_names: string list
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    if num_channels is None:
        if input.num_filters is not None:
            num_channels = input.num_filters
        else:
            num_channels = input.size
    assert (batch_norm_type is None) or (batch_norm_type == "batch_norm") or \
           (batch_norm_type == "mkldnn_batch_norm") or \
           (batch_norm_type == "cudnn_batch_norm")

    l = Layer(
        name=name,
        img3D=img3D,
        inputs=Input(
            input.name, image=Image(channels=num_channels), **param_attr.attr),
        active_type=act.name,
        type=LayerType.BATCH_NORM_LAYER,
        batch_norm_type=batch_norm_type,
        bias=ParamAttr.to_bias(bias_attr),
        epsilon=epsilon,
        moving_average_fraction=moving_average_fraction,
        use_global_stats=use_global_stats,
        mean_var_names=mean_var_names,
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

    :param input: The input of this layer.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute
                       for details.
    :type layer_attr: ExtraLayerAttribute
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


@wrap_name_default()
@layer_support()
def row_l2_norm_layer(input, name=None, layer_attr=None):
    """
    A layer for L2-normalization in each row.

    .. math::
       out[i] = \\frac{in[i]} {\\sqrt{\\sum_{k=1}^N in[k]^{2}}}

    where the size of :math:`in` is (batchSize x dataDim) ,
    and the size of :math:`out` is a (batchSize x dataDim) .

    The example usage is:

    .. code-block:: python

       row_l2_norm_layer = row_l2_norm_layer(input=layer)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute
                       for details.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    Layer(
        name=name,
        type=LayerType.ROW_L2_NORM_LAYER,
        inputs=[input.name],
        **ExtraAttr.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.ROW_L2_NORM_LAYER, parents=[input], size=input.size)


@wrap_name_default("addto")
@wrap_act_default(act=LinearActivation())
@wrap_bias_attr_default(has_bias=False)
@layer_support(DROPOUT, ERROR_CLIPPING)
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

    This layer just simply adds all input layers together, then activates the
    sum. All inputs should share the same dimension, which is also the dimension
    of this layer's output.

    There is no weight matrix for each input, because it just a simple add
    operation. If you want a complicated operation before add, please use
    mixed_layer.

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input layers. It could be a LayerOutput or list/tuple of
                 LayerOutput.
    :type input: LayerOutput | list | tuple
    :param act: Activation Type. LinearActivation is the default activation.
    :type act: BaseActivation
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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
@layer_support(DROPOUT, ERROR_CLIPPING)
def concat_layer(input, act=None, name=None, layer_attr=None, bias_attr=None):
    """
    Concatenate all input vectors to one vector.
    Inputs can be a list of LayerOutput or a list of projection.

    The example usage is:

    ..  code-block:: python

        concat = concat_layer(input=[layer1, layer2])

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input layers or projections
    :type input: list | tuple | collections.Sequence
    :param act: Activation type. IdentityActivation is the default activation.
    :type act: BaseActivation
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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

    layer = Layer(
        name=name,
        type=layer_type,
        inputs=[x.name for x in input] if is_concat_layer else input,
        active_type=act.name,
        bias=ParamAttr.to_bias(bias_attr),
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    sz = layer.config.size

    return LayerOutput(
        name,
        layer_type=layer_type,
        parents=input if is_concat_layer else [x.origin for x in input],
        activation=act,
        size=sz)


@wrap_name_default("seqconcat")
@wrap_act_default(act=IdentityActivation())
@wrap_bias_attr_default(has_bias=False)
@layer_support(DROPOUT, ERROR_CLIPPING)
def seq_concat_layer(a, b, act=None, name=None, layer_attr=None,
                     bias_attr=None):
    """
    Concatenate sequence a and sequence b.

    Inputs:
      - a = [a1, a2, ..., am]
      - b = [b1, b2, ..., bn]

    Output: [a1, ..., am, b1, ..., bn]

    Note that the above computation is for one sample. Multiple samples are
    processed in one batch.

    The example usage is:

    ..  code-block:: python

        concat = seq_concat_layer(a=layer1, b=layer2)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param a: The first input sequence layer
    :type a: LayerOutput
    :param b: The second input sequence layer
    :type b: LayerOutput
    :param act: Activation type. IdentityActivation is the default activation.
    :type act: BaseActivation
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
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
    The memory takes a layer's output at previous time step as its own output.

    If boot_bias, the activation of the bias is the initial value of the memory.

    If boot_with_const_id is set, then the memory's output at the first time step
    is a IndexSlot, the Arguments.ids()[0] is this :code:`cost_id`.

    If boot_layer is specified, the memory's output at the first time step will
    be the boot_layer's output.

    In other case, the default memory's output at the first time step is zero.

    .. code-block:: python

       mem = memory(size=256, name='state')
       state = fc_layer(input=mem, size=256, name='state')

    If you do not want to specify the name, you can also use set_input()
    to specify the layer to be remembered as the following:

    .. code-block:: python

       mem = memory(size=256)
       state = fc_layer(input=mem, size=256)
       mem.set_input(mem)

    :param name: The name of the layer which this memory remembers.
                 If name is None, user should call set_input() to specify the
                 name of the layer which this memory remembers.
    :type name: basestring
    :param size: The dimensionality of memory.
    :type size: int
    :param memory_name: The name of the memory. It is ignored when name is provided.
    :type memory_name: basestring
    :param is_seq: DEPRECATED. is sequence for boot_layer
    :type is_seq: bool
    :param boot_layer: This parameter specifies memory's output at the first time
                       step and the output is boot_layer's output.
    :type boot_layer: LayerOutput | None
    :param boot_bias: The bias attribute of memory's output at the first time step.
                      If the parameter is set to False or an object whose type is not
                      ParameterAttribute, no bias is defined. If the parameter is set
                      to True, the bias is initialized to zero.
    :type boot_bias: ParameterAttribute | None
    :param boot_bias_active_type: Activation type for memory's bias at the first time
                                  step. LinearActivation is the default activation.
    :type boot_bias_active_type: BaseActivation
    :param boot_with_const_id: This parameter specifies memory's output at the first
                               time step and the output is an index.
    :type boot_with_const_id: int
    :return: LayerOutput object.
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
@wrap_act_default(param_names=['gate_act'], act=SigmoidActivation())
@wrap_act_default(param_names=['state_act'], act=TanhActivation())
@wrap_act_default(act=TanhActivation())
@wrap_name_default('lstm_step')
@layer_support()
def lstm_step_layer(input,
                    state,
                    size=None,
                    act=None,
                    name=None,
                    gate_act=None,
                    state_act=None,
                    bias_attr=None,
                    layer_attr=None):
    """
    LSTM Step Layer. This function is used only in recurrent_group.
    The lstm equations are shown as follows.

    ..  math::

        i_t & = \\sigma(W_{x_i}x_{t} + W_{h_i}h_{t-1} + W_{c_i}c_{t-1} + b_i)

        f_t & = \\sigma(W_{x_f}x_{t} + W_{h_f}h_{t-1} + W_{c_f}c_{t-1} + b_f)

        c_t & = f_tc_{t-1} + i_t tanh (W_{x_c}x_t+W_{h_c}h_{t-1} + b_c)

        o_t & = \\sigma(W_{x_o}x_{t} + W_{h_o}h_{t-1} + W_{c_o}c_t + b_o)

        h_t & = o_t tanh(c_t)


    The input of lstm step is :math:`Wx_t + Wh_{t-1}`, and user should use
    :code:`mixed_layer` and :code:`full_matrix_projection` to calculate these
    input vectors.

    The state of lstm step is :math:`c_{t-1}`. And lstm step layer will do

    ..  math::

        i_t = \\sigma(input + W_{ci}c_{t-1} + b_i)

        ...


    This layer has two outputs. The default output is :math:`h_t`. The other
    output is :math:`o_t`, whose name is 'state' and users can use
    :code:`get_output_layer` to extract this output.

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param size: The dimension of this layer's output, which must be
                 equal to the dimension of the state.
    :type size: int
    :param input: The input of this layer.
    :type input: LayerOutput
    :param state: The state of the LSTM unit.
    :type state: LayerOutput
    :param act: Activation type. TanhActivation is the default activation.
    :type act: BaseActivation
    :param gate_act: Activation type of the gate. SigmoidActivation is the
                     default activation.
    :type gate_act: BaseActivation
    :param state_act: Activation type of the state. TanhActivation is the
                      default activation.
    :type state_act: BaseActivation
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert size is None or state.size == size
    size = state.size
    Layer(
        name=name,
        type=LayerType.LSTM_STEP_LAYER,
        active_type=act.name,
        active_gate_type=gate_act.name,
        active_state_type=state_act.name,
        bias=ParamAttr.to_bias(bias_attr),
        size=state.size,
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

    :param input: The input of this layer, whose dimension can be divided by 3.
    :type input: LayerOutput
    :param output_mem: A memory which memorizes the output of this layer at previous
                       time step.
    :type output_mem: LayerOutput
    :param size: The dimension of this layer's output. If it is not set or set to None,
                 it will be set to one-third of the dimension of the input automatically.
    :type size: int
    :param act: Activation type of this layer's output. TanhActivation
                is the default activation.
    :type act: BaseActivation
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param gate_act: Activation type of this layer's two gates. SigmoidActivation is
                     the default activation.
    :type gate_act: BaseActivation
    :param bias_attr: The parameter attribute for bias. If this parameter is set to
                      False or an object whose type is not ParameterAttribute, no bias
                      is defined. If this parameter is set to True,
                      the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for details.
    :type layer_attr: ExtraLayerAttribute
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
    GRU Step Layer, which is realized using PaddlePaddle API. It supports ERROR_CLIPPING
    and DROPOUT.

    :param input: The input of this layer, whose dimensionality can be divided by 3.
    :param output_mem: A memory which memorizes the output of this layer at previous
                       time step.
    :type output_mem: LayerOutput
    :param size: The dimension of this layer's output. If it is not set or set to None,
                 it will be set to one-third of the dimension of the input automatically.
    :type size: int
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param act: Activation type of this layer's output. TanhActivation
                is the default activation.
    :type act: BaseActivation
    :param gate_act: Activation type of this layer's two gates. SigmoidActivation
                     is the default activation.
    :type gate_act: BaseActivation
    :param bias_attr: The parameter attribute for bias. If this parameter is set to
                      False or an object whose type is not ParameterAttribute, no bias
                      is defined. If this parameter is set to True,
                      the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if input.size % 3 != 0:
        raise ValueError("GruStep input size must be divided by 3")
    if size is None:
        size = input.size / 3

    if bias_attr and bias_attr.attr.get("parameter_name", None) is not None:
        raise ValueError("You should not specify the field `name` in bias_attr."
                         " Otherwise, the three biases, which correponding to "
                         " the two gates and the mixed layer for computing Wx+b"
                         ", will share the same parameter matrix unexpectedly.")

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

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input layer. And this layer should contain
                   multiple outputs.
    :type input: LayerOutput
    :param arg_name: The name of the output to be extracted from the input layer.
    :type arg_name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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


    :param input: The input of this layer.
    :type input: LayerOutput
    :param act: Activation type. TanhActivation is the default activation.
    :type act: BaseActivation
    :param bias_attr: The parameter attribute for bias. If this parameter is set to
                      False or an object whose type is not ParameterAttribute,
                      no bias is defined. If the parameter is set to True,
                      the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param param_attr: The parameter attribute. See ParameterAttribute for
                       details.
    :type param_attr: ParameterAttribute
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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
    and can be a sequence or non-sequence.
    :param size: DEPRECATED
    :param is_seq: DEPRECATED
    """

    def __init__(self, input, is_seq=False, size=None):
        assert isinstance(input, LayerOutput)
        self.input = input
        assert input.size is not None
        if size is not None:
            assert input.size == size


def SubsequenceInput(input):
    """
    DEPRECATED.
    Input sequence has sub-sequence, used in recurrent_group.

    The example usage is:

    .. code-block:: python

       input = SubsequenceInput(layer)
    """
    return input


@wrap_name_default("recurrent_group")
def recurrent_group(step, input, reverse=False, name=None, targetInlink=None):
    """
    Recurrent layer group is an extremely flexible recurrent unit in
    PaddlePaddle. As long as the user defines the calculation done within a
    time step, PaddlePaddle will iterate such a recurrent calculation over
    sequence input. This is useful for attention-based models, or Neural
    Turning Machine like models.

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

    :param step: A step function which takes the input of recurrent_group as its own
                 input and returns values as recurrent_group's output every time step.

                 The recurrent group scatters a sequence into time steps. And
                 for each time step, it will invoke step function, and return
                 a time step result. Then gather outputs of each time step into
                 layer group's output.

    :type step: callable

    :param name: The recurrent_group's name. It is optional.
    :type name: basestring

    :param input: Input links array.

                  LayerOutput will be scattered into time steps.
                  SubsequenceInput will be scattered into sequence steps.
                  StaticInput will be imported to each time step, and doesn't change
                  over time. It's a mechanism to access layer outside step function.

    :type input: LayerOutput | StaticInput | SubsequenceInput | list | tuple

    :param reverse: If reverse is set to True, the recurrent unit will process the
                    input sequence in a reverse order.
    :type reverse: bool

    :param targetInlink: DEPRECATED.
                         The input layer which share info with layer group's output

                         Param input specifies multiple input layers. For
                         SubsequenceInput inputs, config should assign one input
                         layer that share info(the number of sentences and the number
                         of words in each sentence) with all layer group's outputs.
                         targetInlink should be one of the layer group's input.

    :type targetInlink: LayerOutput | SubsequenceInput

    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    model_type('recurrent_nn')

    if isinstance(input, LayerOutput) or isinstance(input, StaticInput):
        input = [input]
    assert isinstance(input, collections.Sequence)

    def is_in_links(x):
        return isinstance(x, LayerOutput)

    in_links = filter(is_in_links, input)

    RecurrentLayerGroupWithoutOutLinksBegin(
        name=name,
        in_links=map(lambda x: x.name, in_links),
        seq_reversed=reverse)
    in_args = []
    for each_input in input:
        if isinstance(each_input, StaticInput):  # StaticInput
            mem_name = "__%s_memory__" % each_input.input.name
            mem = memory(
                name=None,
                size=each_input.input.size,
                boot_layer=each_input.input)
            mem.set_input(mem)
            in_args.append(mem)
        else:
            in_args.append(each_input)

    layer_outs = step(*in_args)

    if isinstance(layer_outs, LayerOutput):
        layer_outs = [layer_outs]

    for layer_out in layer_outs:
        assert isinstance(
            layer_out, LayerOutput
        ), "Type of step function's return value must be LayerOutput."
        layer_out.reverse = reverse
        RecurrentLayerGroupSetOutLink(layer_out.name)

    RecurrentLayerGroupEnd(name=name)

    for layer_out in layer_outs:
        # The previous full_name is the name inside the recurrent group.
        # We need a full_name outside the recurrent group.
        layer_out.full_name = MakeLayerNameInSubmodel(layer_out.name)

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
        if isinstance(input, LayerOutput):
            input = [input]
        elif isinstance(input, collections.Sequence):
            input = list(input)
            if len(input) > 1:
                logger.info(
                    ("More than one layers inside the recurrent_group "
                     "are returned as outputs of the entire recurrent_group "
                     "PLEASE garantee the first output is probability of "
                     "the predicted next word."))

        return [maxid_layer(
            input=input[0], name='__beam_search_predict__')] + (
                input[1:] if len(input) > 1 else [])

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

    :param input: The input of this layer.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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
def dot_prod_layer(input1, input2, name=None, layer_attr=None):
    """
    A layer for computing the dot product of two vectors.

    The example usage is:

    .. code-block:: python

        dot_prod = dot_prod_layer(input1=vec1, input2=vec2)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input1: The first input layer.
    :type input1: LayerOutput
    :param input2: The second input layer.
    :type input2: LayerOutput
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input1, LayerOutput)
    assert isinstance(input2, LayerOutput)
    assert input1.size == input2.size, ("Two inputs should have the same size.")

    l = Layer(
        name=name,
        type=LayerType.DOT_PROD_LAYER,
        inputs=[input1.name, input2.name],
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        layer_type=LayerType.DOT_PROD_LAYER,
        parents=[input1, input2],
        size=l.config.size)


@wrap_name_default()
def out_prod_layer(input1, input2, name=None, layer_attr=None):
    """
    A layer for computing the outer product of two vectors
    The result is a matrix of size(input1) x size(input2)

    The example usage is:

    .. code-block:: python

       out_prod = out_prod_layer(input1=vec1, input2=vec2)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input1: The first input layer.
    :type input: LayerOutput
    :param input2: The second input layer.
    :type input2: LayerOutput
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param eos_id: End id of sequence
    :type eos_id: int
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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

        generated_word_embedding = GeneratedInput(
                               size=target_dictionary_dim,
                               embedding_name="target_language_embedding",
                               embedding_size=word_vector_dim)

        beam_gen = beam_search(name="decoder",
                               step=rnn_step,
                               input=[StaticInput(encoder_last),
                                      generated_word_embedding],
                               bos_id=0,
                               eos_id=1,
                               beam_size=5)

    Please see the following demo for more details:

    - machine translation : demo/seqToseq/translation/gen.conf \
                            demo/seqToseq/seqToseq_net.py

    :param name: The name of the recurrent unit that is responsible for
                 generating sequences. It is optional.
    :type name: basestring
    :param step: A callable function that defines the calculation in a time
                 step, and it is applied to sequences with arbitrary length by
                 sharing a same set of weights.

                 You can refer to the first parameter of recurrent_group, or
                 demo/seqToseq/seqToseq_net.py for more details.
    :type step: callable
    :param input: Input data for the recurrent unit, which should include the
                  previously generated words as a GeneratedInput object.
                  In beam_search, none of the input's type should be LayerOutput.
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
        assert not isinstance(each_input, LayerOutput), (
            "in beam_search, "
            "none of the input should has a type of LayerOutput.")
        if isinstance(each_input, BaseGeneratedInput):
            assert generated_input_index == -1, ("recurrent_group accepts "
                                                 "only one GeneratedInput.")
            generated_input_index = i

        else:
            real_input.append(each_input)

    assert generated_input_index != -1, "No GeneratedInput is given."

    gipt = input[generated_input_index]

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

        eos_layer(input=predict[0], eos_id=eos_id, name=eos_name)
        return predict

    return recurrent_group(
        step=__real_step__, input=real_input, reverse=False, name=name)


def __cost_input__(input, label, weight=None):
    """
    inputs and parents for cost layers.
    """
    if isinstance(input, LayerOutput):
        input = [input]
    if isinstance(label, LayerOutput):
        label = [label]
    ipts = [Input(ipt.name) for ipt in (input + label)]
    parents = [ipt for ipt in (input + label)]
    if weight is not None:
        assert weight.size == 1
        ipts.append(Input(weight.name))
        parents.append(weight)
    return ipts, parents


@wrap_name_default()
@layer_support()
def square_error_cost(input,
                      label,
                      weight=None,
                      name=None,
                      coeff=1.0,
                      layer_attr=None):
    """
    sum of square error cost:

    ..  math::

        cost = \\sum_{i=1}^N(t_i-y_i)^2

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The first input layer.
    :type input: LayerOutput
    :param label: The input label.
    :type label: LayerOutput
    :param weight: The weight layer defines a weight for each sample in the
                   mini-batch. It is optional.
    :type weight: LayerOutput
    :param coeff: The weight of the gradient in the back propagation.
                  1.0 is the default value.
    :type coeff: float
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    ipts, parents = __cost_input__(input, label, weight)

    Layer(
        inputs=ipts,
        type="square_error",
        name=name,
        coeff=coeff,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(name, LayerType.COST, parents=parents, size=1)


regression_cost = square_error_cost


@wrap_name_default("cost")
@layer_support()
def classification_cost(input,
                        label,
                        weight=None,
                        name=None,
                        evaluator=classification_error_evaluator,
                        layer_attr=None,
                        coeff=1.):
    """
    classification cost Layer.

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The first input layer.
    :type input: LayerOutput
    :param label: The input label.
    :type label: LayerOutput
    :param weight: The weight layer defines a weight for each sample in the
                   mini-batch. It is optional.
    :type weight: LayerOutput
    :param evaluator: Evaluator method. classification_error_evaluator is the default.
    :type evaluator: Evaluator method
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :param coeff: The weight of the gradient in the back propagation.
                  1.0 is the default value.
    :type coeff: float
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
        coeff=coeff,
        **ExtraLayerAttribute.to_kwargs(layer_attr))

    def __add_evaluator__(e):
        assert callable(e)
        assert hasattr(e, 'is_evaluator')
        assert isinstance(e.is_evaluator, bool)
        assert e.is_evaluator
        assert hasattr(e, "for_classification")
        assert isinstance(e.for_classification, bool)
        assert e.for_classification

        e(name=e.__name__, input=input, label=label, weight=weight)

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
    supports GPU mode.

    The example usage is:

    .. code-block:: python

       op = conv_operator(img=input1,
                          filter=input2,
                          filter_size=3,
                          num_filters=64,
                          num_channels=64)

    :param img: The input image.
    :type img: LayerOutput
    :param filter: The input filter.
    :type filter: LayerOutput
    :param filter_size: The dimension of the filter kernel on the x axis.
    :type filter_size: int
    :param filter_size_y: The dimension of the filter kernel on the y axis.
                          If the parameter is not set or set to None, it will
                          set to 'filter_size' automatically.
    :type filter_size_y: int
    :param num_filters: The number of the output channels.
    :type num_filters: int
    :param num_channels: The number of the input channels. If the parameter is not set
                         or set to None, it will be automatically set to the channel
                         number of the 'img'.
    :type num_channels: int
    :param stride: The stride on the x axis.
    :type stride: int
    :param stride_y: The stride on the y axis. If the parameter is not set or
                     set to None, it will be set to 'stride' automatically.
    :type stride_y: int
    :param padding: The padding size on the x axis.
    :type padding: int
    :param padding_y: The padding size on the y axis. If the parameter is not set
                      or set to None, it will be set to 'padding' automatically.
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
    assert filter.size is not None

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
    Different from img_conv_layer and conv_op, conv_projection is a Projection,
    which can be used in mixed_layer and concat_layer. It uses cudnn to implement
    convolution and only supports GPU mode.

    The example usage is:

    .. code-block:: python

       proj = conv_projection(input=input1,
                              filter_size=3,
                              num_filters=64,
                              num_channels=64)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param filter_size: The dimensions of the filter kernel. If the parameter is
                        set to one integer, the two dimensions on x and y axises
                        will be same when filter_size_y is not set. If it is set
                        to a list, the first element indicates the dimension on
                        the x axis, and the second is used to specify the dimension
                        on the y axis when filter_size_y is not provided.
    :type filter_size: int | tuple | list
    :param filter_size_y: The dimension of the filter kernel on the y axis. If the parameter
                          is not set, it will be set automatically according to filter_size.
    :type filter_size_y: int
    :param num_filters: The number of filters.
    :type num_filters: int
    :param num_channels: The number of the input channels.
    :type num_channels: int
    :param stride: The strides. If the parameter is set to one integer, the strides
                   on x and y axises will be same when stride_y is not set. If it is
                   set to a list, the first element indicates the stride on the x axis,
                   and the second is used to specify the stride on the y axis when
                   stride_y is not provided.
    :type stride: int | tuple | list
    :param stride_y: The stride on the y axis.
    :type stride_y: int
    :param padding: The padding sizes. If the parameter is set to one integer, the padding
                    sizes on x and y axises will be same when padding_y is not set. If it
                    is set to a list, the first element indicates the padding size on the
                    x axis, and the second is used to specify the padding size on the y axis
                    when padding_y is not provided.
    :type padding: int | tuple | list
    :param padding_y: The padding size on the y axis.
    :type padding_y: int
    :param groups: The group number.
    :type groups: int
    :param param_attr: The parameter attribute of the convolution. See ParameterAttribute for
                       details.
    :type param_attr: ParameterAttribute
    :param trans: Whether it is ConvTransProjection or ConvProjection
    :type trans: bool
    :return: A Projection Object.
    :rtype: ConvTransProjection | ConvProjection
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
    and pad_w. pad_c, pad_h, pad_w specify the size in the corresponding
    dimension. And the input data shape is NCHW.

    For example, pad_c=[2,3] means padding 2 zeros before the input data
    and 3 zeros after the input data in the channel dimension. pad_h means
    padding zeros in the height dimension. pad_w means padding zeros in the
    width dimension.

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

    :param input: The input of this layer.
    :type input: LayerOutput
    :param pad_c: The padding size in the channel dimension.
    :type pad_c: list | None
    :param pad_h: The padding size in the height dimension.
    :type pad_h: list | None
    :param pad_w: The padding size in the width dimension.
    :type pad_w: list | None
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :param name: The name of this layer. It is optional.
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
    This layer performs cyclic convolution on two inputs. For example:
      - a[in]: contains M elements.
      - b[in]: contains N elements (N should be odd).
      - c[out]: contains M elements.

    .. math::

        c[i] = \sum_{j=-(N-1)/2}^{(N-1)/2}a_{i+j} * b_{j}

    In this formula:
     - a's index is computed modulo M. When it is negative, then get item from
       the right side (which is the end of array) to the left.
     - b's index is computed modulo N. When it is negative, then get item from
       the right size (which is the end of array) to the left.

    The example usage is:

    .. code-block:: python

       conv_shift = conv_shift_layer(a=layer1, b=layer2)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param a: The first input of this layer.
    :type a: LayerOutput
    :param b: The second input of this layer.
    :type b: LayerOutput
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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
    This layer performs tensor operation on two inputs.
    For example:

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

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param a: The first input of this layer.
    :type a: LayerOutput
    :param b: The second input of this layer.
    :type b: LayerOutput
    :param size: The dimension of this layer.
    :type size: int
    :param act: Activation type. LinearActivation is the default activation.
    :type act: BaseActivation
    :param param_attr: The parameter attribute. See ParameterAttribute for
                       details.
    :type param_attr: ParameterAttribute
    :param bias_attr: The parameter attribute for bias. If this parameter is set to
                      False or an object whose type is not ParameterAttribute,
                      no bias is defined. If this parameter is set to True,
                      the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute | None
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
@layer_support(DROPOUT, ERROR_CLIPPING)
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
    of this layer can be sparse. It requires an additional input to indicate
    several selected columns for output. If the selected columns is not
    specified, selective_fc_layer acts exactly like fc_layer.

    The simple usage is:

    .. code-block:: python

       sel_fc = selective_fc_layer(input=input, size=128, act=TanhActivation())

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput | list | tuple
    :param select: The layer to select columns to output. It should be a sparse
                   binary matrix, and is treated as the mask of selective fc. If
                   it is not set or set to None, selective_fc_layer acts exactly
                   like fc_layer.
    :type select: LayerOutput
    :param size: The dimension of this layer, which should be equal to that of
                 the layer 'select'.
    :type size: int
    :param act: Activation type. TanhActivation is the default activation.
    :type act: BaseActivation
    :param pass_generation: The flag which indicates whether it is during generation.
    :type pass_generation: bool
    :param has_selected_colums: The flag which indicates whether the parameter 'select'
                                has been set. True is the default.
    :type has_selected_colums: bool
    :param mul_ratio: A ratio helps to judge how sparse the output is and determine
                      the computation method for speed consideration.
    :type mul_ratio: float
    :param param_attr: The parameter attribute. See ParameterAttribute for
                       details.
    :type param_attr: ParameterAttribute
    :param bias_attr: The parameter attribute for bias. If this parameter is set to
                      False or an object whose type is not ParameterAttribute,
                      no bias is defined. If this parameter is set to True,
                      the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute | None
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
            if "parameter_name" in param_attr.attr and len(input) > 1:
                logger.fatal(
                    "When the name field of param_attr is manually specified "
                    "and the input is a list, the param_attr should also be a "
                    "list with each item being the param_attr for each input "
                    "item. If only one named param_attr is provided, all the "
                    "input items would share this parameter.")
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
    A layer for sampling id from a multinomial distribution from the input layer.
    Sampling one id for one sample.

    The simple usage is:

    .. code-block:: python

       samping_id = sampling_id_layer(input=input)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
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
    This layer for applying a slope and an intercept to the input.

    ..  math::
        y = slope * x + intercept

    The simple usage is:

    .. code-block:: python

       scale = slope_intercept_layer(input=input, slope=-1.0, intercept=1.0)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param slope: The scale factor.
    :type slope: float
    :param intercept: The offset.
    :type intercept: float
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
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
    :param size: The dimension of this layer.
    :type size: int
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
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

    The expanding method is the same with ExpandConvLayer, but saved the transposed
    value. After expanding, output.sequenceStartPositions will store timeline.
    The number of time steps is outputH * outputW and the dimension of each
    time step is block_y * block_x * num_channels. This layer can be used after
    convolutional neural network, and before recurrent neural network.

    The simple usage is:

    .. code-block:: python

       block_expand = block_expand_layer(input=layer,
                                         num_channels=128,
                                         stride_x=1,
                                         stride_y=1,
                                         block_x=1,
                                         block_x=3)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param num_channels: The number of input channels. If the parameter is not set or
                         set to None, its actual value will be automatically set to
                         the channels number of the input.
    :type num_channels: int
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
    :param name: The name of this layer. It is optional.
    :type name: basestring.
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
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
    A layer to do max out on convolutional layer output.
      - Input: the output of a convolutional layer.
      - Output: feature map size same as the input's, and its channel number is
        (input channel) / groups.

    So groups should be larger than 1, and the num of channels should be able
    to be devided by groups.

    Reference:
        `Maxout Networks
        <http://www.jmlr.org/proceedings/papers/v28/goodfellow13.pdf>`_
        `Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks
        <https://arxiv.org/pdf/1312.6082v4.pdf>`_


    .. math::

       & out = \max_k (in[n, k, o_c , s])

       & out_{i * s + j} = \max_k in_{  k * o_{c} * s + i * s + j}

       & s = \\frac{input.size}{ num\_channels}

       & o_{c} = \\frac{num\_channels}{groups}

       & 0 \le i < o_{c}

       & 0 \le j < s

       & 0 \le k < groups


    The simple usage is:

    .. code-block:: python

       maxout = maxout_layer(input,
                             num_channels=128,
                             groups=4)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param num_channels: The number of input channels. If the parameter is not set or
                         set to None, its actual value will be automatically set to
                         the channels number of the input.
    :type num_channels: int
    :param groups: The group number of input layer.
    :type groups: int
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
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
    classication task. e.g. sequence labeling problems where the
    alignment between the inputs and the target labels is unknown.

    Reference:
        `Connectionist Temporal Classification: Labelling Unsegmented Sequence Data
        with Recurrent Neural Networks
        <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf>`_

    Note:
        Considering the 'blank' label needed by CTC, you need to use (num_classes + 1)
        as the size of the input, where num_classes is the category number.
        And the 'blank' is the last category index. So the size of 'input' layer (e.g.
        fc_layer with softmax activation) should be (num_classes + 1). The size of
        ctc_layer should also be (num_classes + 1).

    The example usage is:

    .. code-block:: python

      ctc = ctc_layer(input=input,
                      label=label,
                      size=9055,
                      norm_by_times=True)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param label: The input label.
    :type label: LayerOutput
    :param size: The dimension of this layer, which must be equal to (category number + 1).
    :type size: int
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param norm_by_times: Whether to do normalization by times. False is the default.
    :type norm_by_times: bool
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
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
    <https://github.com/baidu-research/warp-ctc>`_ library, which is used in
    `Deep Speech 2: End-toEnd Speech Recognition in English and Mandarin
    <https://arxiv.org/pdf/1512.02595v1.pdf>`_, to compute Connectionist Temporal
    Classification (CTC) loss. Besides, another `warp-ctc repository
    <https://github.com/gangliao/warp-ctc>`_ , which is forked from
    the official one, is maintained to enable more compiling options. During the
    building process, PaddlePaddle will clone the source codes, build and
    install it to :code:`third_party/install/warpctc` directory.

    Reference:
        `Connectionist Temporal Classification: Labelling Unsegmented Sequence Data
        with Recurrent Neural Networks
        <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf>`_

    Note:
        - Let num_classes represents the category number. Considering the 'blank'
          label needed by CTC, you need to use (num_classes + 1) as the size of
          warp_ctc layer.
        - You can set 'blank' to any value ranged in [0, num_classes], which
          should be consistent with those used in your labels.
        - As a native 'softmax' activation is interated to the warp-ctc library,
          'linear' activation is expected to be used instead in the 'input' layer.

    The example usage is:

    .. code-block:: python

      ctc = warp_ctc_layer(input=input,
                           label=label,
                           size=1001,
                           blank=1000,
                           norm_by_times=False)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param label: The input label.
    :type label: LayerOutput
    :param size: The dimension of this layer, which must be equal to (category number + 1).
    :type size: int
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param blank: The 'blank' label used in ctc.
    :type blank: int
    :param norm_by_times: Whether to do normalization by times. False is the default.
    :type norm_by_times: bool
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
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
              coeff=1.0,
              layer_attr=None):
    """
    A layer for calculating the cost of sequential conditional random
    field model.

    The example usage is:

    .. code-block:: python

      crf = crf_layer(input=input,
                      label=label,
                      size=label_dim)

    :param input: The first input layer.
    :type input: LayerOutput
    :param label: The input label.
    :type label: LayerOutput
    :param size: The category number.
    :type size: int
    :param weight: The weight layer defines a weight for each sample in the
                   mini-batch. It is optional.
    :type weight: LayerOutput
    :param param_attr: The parameter attribute. See ParameterAttribute for
                       details.
    :type param_attr: ParameterAttribute
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param coeff: The weight of the gradient in the back propagation.
                  1.0 is the default value.
    :type coeff: float
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
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
        coeff=coeff,
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
    If the input 'label' is provided, it is treated as the ground-truth label, and
    this layer will also calculate error. output.value[i] is 1 for an incorrect
    decoding and 0 for the correct.

    The example usage is:

    .. code-block:: python

      crf_decoding = crf_decoding_layer(input=input,
                                        size=label_dim)

    :param input: The first input layer.
    :type input: LayerOutput
    :param size: The dimension of this layer.
    :type size: int
    :param label: The input label.
    :type label: LayerOutput | None
    :param param_attr: The parameter attribute. See ParameterAttribute for
                       details.
    :type param_attr: ParameterAttribute
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
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


"""
Following are cost Layers.
"""


@wrap_bias_attr_default(has_bias=True)
@wrap_param_attr_default()
@wrap_name_default()
@layer_support()
def nce_layer(input,
              label,
              num_classes=None,
              param_attr=None,
              weight=None,
              num_neg_samples=10,
              neg_distribution=None,
              name=None,
              bias_attr=None,
              layer_attr=None):
    """
    Noise-contrastive estimation.

    Reference:
        `A fast and simple algorithm for training neural probabilistic language
        models. <https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf>`_

    The example usage is:

    .. code-block:: python

       cost = nce_layer(input=[layer1, layer2], label=layer2,
                        param_attr=[attr1, attr2], weight=layer3,
                        num_classes=3, neg_distribution=[0.1,0.3,0.6])

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The first input of this layer.
    :type input: LayerOutput | list | tuple | collections.Sequence
    :param label: The input label.
    :type label: LayerOutput
    :param weight: The weight layer defines a weight for each sample in the
                   mini-batch. It is optional.
    :type weight: LayerOutput
    :param num_classes: The number of classes.
    :type num_classes: int
    :param act: Activation type. SigmoidActivation is the default activation.
    :type act: BaseActivation
    :param param_attr: The parameter attribute. See ParameterAttribute for
                       details.
    :type param_attr: ParameterAttribute
    :param num_neg_samples: The number of sampled negative labels. 10 is the
                            default value.
    :type num_neg_samples: int
    :param neg_distribution: The discrete noisy distribution over the output
                             space from which num_neg_samples negative labels
                             are sampled. If this parameter is not set, a
                             uniform distribution will be used. A user-defined
                             distribution is a list whose length must be equal
                             to the num_classes. Each member of the list defines
                             the probability of a class given input x.
    :type neg_distribution: list | tuple | collections.Sequence | None
    :param bias_attr: The parameter attribute for bias. If this parameter is set to
                      False or an object whose type is not ParameterAttribute,
                      no bias is defined. If this parameter is set to True,
                      the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
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

    assert isinstance(label, LayerOutput)
    assert label.layer_type == LayerType.DATA
    if num_classes is None:
        num_classes = label.size
    if neg_distribution is not None:
        assert isinstance(neg_distribution, collections.Sequence)
        assert len(neg_distribution) == num_classes
        assert abs(sum(neg_distribution) - 1.0) < 1e-5

    ipts_for_layer = []
    parents = []
    for each_input, attr in zip(input, param_attr):
        assert isinstance(each_input, LayerOutput)
        ipts_for_layer.append(Input(each_input.name, **attr.attr))
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
        active_type=SigmoidActivation().name,
        num_neg_samples=num_neg_samples,
        inputs=ipts_for_layer,
        bias=ParamAttr.to_bias(bias_attr),
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name,
        LayerType.NCE_LAYER,
        parents=parents,
        size=l.config.size,
        activation=SigmoidActivation())


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
    A cost Layer for learning to rank using gradient descent.

    Reference:
        `Learning to Rank using Gradient Descent
        <http://research.microsoft.com/en-us/um/people/cburges/papers/ICML_ranking.pdf>`_

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

    The example usage is:

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
    :param weight: The weight layer defines a weight for each sample in the
                   mini-batch. It is optional.
    :type weight: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param coeff: The weight of the gradient in the back propagation.
                  1.0 is the default value.
    :type coeff: float
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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

    The example usage is:

    .. code-block:: python

      cost = lambda_cost(input=input,
                         score=score,
                         NDCG_num=8,
                         max_sort_size=-1)

    :param input: The first input of this layer, which is often a document
                  samples list of the same query and whose type must be sequence.
    :type input: LayerOutput
    :param score: The scores of the samples.
    :type input: LayerOutput
    :param NDCG_num: The size of NDCG (Normalized Discounted Cumulative Gain),
                     e.g., 5 for NDCG@5. It must be less than or equal to the
                     minimum size of the list.
    :type NDCG_num: int
    :param max_sort_size: The size of partial sorting in calculating gradient. If
                          max_sort_size is equal to -1 or greater than the number
                          of the samples in the list, then the algorithm will sort
                          the entire list to compute the gradient. In other cases,
                          max_sort_size must be greater than or equal to NDCG_num.
    :type max_sort_size: int
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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

    The example usage is:

    .. code-block:: python

       cost = cross_entropy(input=input_layer,
                            label=label_layer)

    :param input: The first input layer.
    :type input: LayerOutput.
    :param label: The input label.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param coeff: The weight of the gradient in the back propagation.
                  1.0 is the default value.
    :type coeff: float
    :param weight: The weight layer defines a weight for each sample in the
                   mini-batch. It is optional.
    :type weight: LayerOutout
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
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

    The example usage is:

    .. code-block:: python

       cost = cross_entropy_with_selfnorm(input=input_layer,
                                          label=label_layer)

    :param input: The first input layer.
    :type input: LayerOutput
    :param label: The input label.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param coeff: The weight of the gradient in the back propagation.
                  1.0 is the default value.
    :type coeff: float
    :param softmax_selfnorm_alpha: The scale factor affects the cost.
    :type softmax_selfnorm_alpha: float
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
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
    A loss layer which calculates the sum of the input as loss.

    The example usage is:

    .. code-block:: python

       cost = sum_cost(input=input_layer)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
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
def huber_regression_cost(input,
                          label,
                          name=None,
                          delta=1.0,
                          coeff=1.0,
                          layer_attr=None):
    """
    In statistics, the Huber loss is a loss function used in robust regression,
    that is less sensitive to outliers in data than the squared error loss.
    Given a prediction f(x), a label y and :math:`\delta`, the loss function
    is defined as:

    .. math::

       loss = 0.5*(y-f(x))^{2}, | y-f(x) | < \delta

       loss = \delta | y-f(x) | - 0.5 \delta ^2, otherwise

    The example usage is:

    .. code-block:: python

       cost = huber_regression_cost(input=input_layer, label=label_layer)

    :param input: The first input layer.
    :type input: LayerOutput
    :param label: The input label.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param delta: The difference between the observed and predicted values.
    :type delta: float
    :param coeff: The weight of the gradient in the back propagation.
                  1.0 is the default value.
    :type coeff: float
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput.
    """
    assert isinstance(input, LayerOutput)
    Layer(
        name=name,
        type=LayerType.HUBER_REGRESSION,
        inputs=[input.name, label.name],
        delta=delta,
        coeff=coeff,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.HUBER_REGRESSION, parents=[input, label], size=1)


@wrap_name_default()
@layer_support()
def huber_classification_cost(input,
                              label,
                              name=None,
                              coeff=1.0,
                              layer_attr=None):
    """
    For classification purposes, a variant of the Huber loss called modified Huber
    is sometimes used. Given a prediction f(x) (a real-valued classifier score) and
    a true binary class label :math:`y\in \{-1, 1 \}`, the modified Huber
    loss is defined as:

    .. math:

       loss = \max ( 0, 1-yf(x) )^2, yf(x) \geq -1

       loss = -4yf(x), otherwise

    The example usage is:

    .. code-block:: python

       cost = huber_classification_cost(input=input_layer, label=label_layer)

    :param input: The first input layer.
    :type input: LayerOutput
    :param label: The input label.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param coeff: The weight of the gradient in the back propagation.
                  1.0 is the default value.
    :type coeff: float
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput)
    if input.size is not None:
        assert input.size == 1
    Layer(
        name=name,
        type=LayerType.HUBER_CLASSIFICATION,
        inputs=[input.name, label.name],
        coeff=coeff,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.HUBER_CLASSIFICATION, parents=[input, label], size=1)


@wrap_name_default()
@layer_support()
def multi_binary_label_cross_entropy(input,
                                     label,
                                     name=None,
                                     coeff=1.0,
                                     layer_attr=None):
    """
    A loss layer for multi binary label cross entropy.

    The example usage is:

    .. code-block:: python

       cost = multi_binary_label_cross_entropy(input=input_layer,
                                               label=label_layer)

    :param input: The first input layer.
    :type input: LayerOutput
    :param label: The input label.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param coeff: The weight of the gradient in the back propagation.
                  1.0 is the default value.
    :type coeff: float
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    if input.activation is None or \
            not isinstance(input.activation, SigmoidActivation):
        logger.log(logging.WARN,
                   ("%s is not a recommended activation for "
                    "multi_binary_label_cross_entropy, sigmoid is better") %
                   repr(input.activation))

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


class BeamInput(object):
    """
    Define the input for cross_entropy_over_beam layer.

    A beam is made up of a triple: the first one is scores over all
    candidates; the second one is indices of top k selected candidates; the
    third one is the index of ground truth, which is also always called
    gold.
    """

    def __init__(self, candidate_scores, selected_candidates, gold):
        assert isinstance(candidate_scores, LayerOutput)
        self.candidate_scores = candidate_scores
        assert candidate_scores.size == 1

        assert isinstance(selected_candidates, LayerOutput)
        self.selected_candidates = selected_candidates

        assert isinstance(gold, LayerOutput)
        self.gold = gold


@wrap_name_default()
@layer_support()
def cross_entropy_over_beam(input, name=None):
    """
    This layer is used in learning to search models, which is to solve complex
    joint prediction problems based on learning to search through a
    problem-defined search space.

    Specifically, the learning to search process for this layer begins with
    searching a target sequence from a nested sequence. In the first search
    step, top beam size sequences with highest scores, indices of these top k
    sequences in the original nested sequence, and the ground truth (also
    called gold) altogether (a triple) make up of the first beam.

    Then, several special positions, for example, start and end positions
    that define meaningful segments are searched. In these searches, top k
    positions with highest scores are selected, and then sequence, starting
    from the selected starts till ends of the sequences (or a fixed position)
    are taken to search next.

    We call the possible top k results returned in one search the beam. This
    search process can be repeated for pre-defined turns and leads to several
    beam expansions.

    Finally, the layer cross_entropy_over_beam takes all the beam expansions
    which contain several candidate targets found along the multi-step search.
    cross_entropy_over_beam calculates cross entropy over the expanded beams
    which all the candidates in the beam as the normalized factor.

    Note that, if gold falls off the beam at search step t, then the cost is
    calculated over the beam at step t.

    This cost layer always works together with kmax_seq_score_layer,
    sub_nested_seq_layer, and sequence_slice_layer to trim the input to form a
    sub-search space.


    The example usage is:

    .. code-block:: python

       cost = cross_entropy_over_beam(input=[
           BeamInput(
               candidate_scores=beam1_candidates,
               selected_candidates=beam1_topk,
               gold=gold1),
           BeamInput(
               candidate_scores=beam2_candidates,
               selected_candidates=beam2_topk,
               gold=gold2),
       ])


    :param input: Input beams for this layer.
    :type input: BeamInput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    if isinstance(input, BeamInput):
        input = [input]
    else:
        assert isinstance(input, list), (
            'input for cross_entropy_over_beam shold be a python list '
            'of BeamInput object.')
        for ipt in input:
            assert isinstance(ipt, BeamInput), (
                'input for cross_entropy_over_beam '
                'should be a BeamInput object.')

    ipts = []
    parents = []
    for beam in input:
        parents += [beam.candidate_scores, beam.selected_candidates, beam.gold]
        ipts += [
            beam.candidate_scores.name, beam.selected_candidates.name,
            beam.gold.name
        ]

    Layer(name=name, type=LayerType.CROSS_ENTROPY_OVER_BEAM, inputs=ipts)
    return LayerOutput(name, LayerType.CROSS_ENTROPY, parents=parents, size=1)


@wrap_name_default()
@layer_support()
def smooth_l1_cost(input, label, name=None, coeff=1.0, layer_attr=None):
    """
    This is a L1 loss but more smooth. It requires that the
    sizes of input and label are equal. The formula is as follows,

    .. math::

        L = \sum_{i} smooth_{L1}(input_i - label_i)

    in which

    .. math::

        smooth_{L1}(x) = \\begin{cases} 0.5x^2& \\text{if}  \\ |x| < 1 \\\\ |x|-0.5& \\text{otherwise} \end{cases}

    Reference:
        `Fast R-CNN
        <https://arxiv.org/pdf/1504.08083v2.pdf>`_

    The example usage is:

    .. code-block:: python

       cost = smooth_l1_cost(input=input_layer,
                             label=label_layer)

    :param input: The input layer.
    :type input: LayerOutput
    :param label: The input label.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param coeff: The weight of the gradient in the back propagation.
                  1.0 is the default value.
    :type coeff: float
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput)
    assert isinstance(label, LayerOutput)
    assert input.size == label.size

    Layer(
        name=name,
        type=LayerType.SMOOTH_L1,
        inputs=[input.name, label.name],
        coeff=coeff,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.SMOOTH_L1, parents=[input, label], size=1)


@wrap_name_default()
def multiplex_layer(input, name=None, layer_attr=None):
    """
    This layer multiplex multiple layers according to the indexes,
    which are provided by the first input layer.
    inputs[0]: the indexes of the layers to form the output of size batchSize.
    inputs[1:N]; the candidate output data.
    For each index i from 0 to batchSize - 1, the i-th row of the output is the
    the same to the i-th row of the (index[i] + 1)-th layer.

    For each i-th row of output:
    .. math::
        y[i][j] = x_{x_{0}[i] + 1}[i][j], j = 0,1, ... , (x_{1}.width - 1)

    where, y is output. :math:`x_{k}` is the k-th input layer and
    :math:`k = x_{0}[i] + 1`.

    The example usage is:

    .. code-block:: python

       maxid = multiplex_layer(input=layers)

    :param input: Input layers.
    :type input: list of LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute.
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert isinstance(input, collections.Sequence)
    assert len(input) > 2, 'multiplex_layer should have more than 2 inputs'
    for i in range(1, len(input)):
        assert isinstance(input[i], LayerOutput)
        assert input[i].size == input[1].size, \
            "All the input layers except the first one should have the same size"

    l = Layer(
        name=name,
        type='multiplex',
        inputs=[x.name for x in input],
        size=input[1].size,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        layer_type=LayerType.MULTIPLEX_LAYER,
        parents=input,
        size=l.config.size)


@wrap_name_default("dropout")
def dropout_layer(input, dropout_rate, name=None):
    """

    The example usage is:

    .. code-block:: python

        dropout = dropout_layer(input=input_layer, dropout_rate=0.5)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param dropout_rate: The probability of dropout.
    :type dropout_rate: float
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    return addto_layer(
        name=name,
        input=input,
        act=LinearActivation(),
        bias_attr=False,
        layer_attr=ExtraAttr(drop_rate=dropout_rate))


@wrap_name_default()
@wrap_act_default(act=LinearActivation())
@wrap_param_attr_default()
@layer_support(DROPOUT)
def row_conv_layer(input,
                   context_len,
                   act=None,
                   name=None,
                   param_attr=None,
                   layer_attr=None):
    """

    The row convolution is called lookahead convolution. It is firstly
    introduced in paper of `Deep Speech 2: End-to-End Speech Recognition
    in English and Mandarin <https://arxiv.org/pdf/1512.02595v1.pdf>`_ .

    The bidirectional RNN that learns representation for a sequence by
    performing a forward and a backward pass through the entire sequence.
    However, unlike unidirectional RNNs, bidirectional RNNs are challenging
    to deploy in an online and low-latency setting. The lookahead convolution
    incorporates information from future subsequences in a computationally
    efficient manner to improve unidirectional RNNs.

    The connection of row convolution is different from the 1D sequence
    convolution. Assumed that, the future context-length is k, that is to say,
    it can get the output at timestep t by using the the input feature from t-th
    timestep to (t+k+1)-th timestep. Assumed that the hidden dim of input
    activations are d, the activations r_t for the new layer at time-step t are:

    .. math::

        r_{t,r} = \sum_{j=1}^{k + 1} {w_{i,j}h_{t+j-1, i}}
                  \quad \\text{for} \quad  (1 \leq i \leq d)

    Note:
        The `context_len` is `k + 1`. That is to say, the lookahead step
        number plus one equals context_len.


    .. code-block:: python

       row_conv = row_conv_layer(input=input_layer, context_len=3)


    :param input: The input of this layer.
    :type input: LayerOutput
    :param context_len: The context length equals the lookahead step number
                        plus one.
    :type context_len: int
    :param act: Activation Type. LinearActivation is the default activation.
    :type act: BaseActivation
    :param param_attr: The parameter attribute. See ParameterAttribute for
                       details.
    :type param_attr: ParameterAttribute
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute | None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput)
    assert context_len > 0, "the context_len must be greatet than 0."

    Layer(
        inputs=[Input(input.name, **param_attr.attr)],
        name=name,
        context_length=context_len,
        type=LayerType.ROW_CONV_LAYER,
        active_type=act.name,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.ROW_CONV_LAYER, input, activation=act, size=input.size)


@layer_support()
@wrap_name_default()
def prelu_layer(input,
                name=None,
                partial_sum=1,
                channel_shared=None,
                num_channels=None,
                param_attr=None,
                layer_attr=None):
    """
    The Parametric Relu activation that actives outputs with a learnable weight.

    Reference:
        `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
        ImageNet Classification <http://arxiv.org/pdf/1502.01852v1.pdf>`_

    .. math::
       z_i &\\quad if \\quad z_i > 0 \\\\
       a_i * z_i  &\\quad \\mathrm{otherwise}

    The example usage is:

    .. code-block:: python

       prelu = prelu_layer(input=layers, partial_sum=1)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param partial_sum: this parameter makes a group of inputs share the same weight.

        - partial_sum = 1, indicates the element-wise activation: each element has a weight.
        - partial_sum = number of elements in one channel, indicates the channel-wise activation, elements in a channel share the same weight.
        - partial_sum = number of outputs, indicates all elements share the same weight.

    :type partial_sum: int
    :param channel_shared: whether or not the parameter are shared across channels.

        - channel_shared = True, we set the partial_sum to the number of outputs.
        - channel_shared = False, we set the partial_sum to the number of elements in one channel.

    :type channel_shared: bool
    :param num_channels: number of input channel.
    :type num_channels: int
    :param param_attr: The parameter attribute. See ParameterAttribute for details.
    :type param_attr: ParameterAttribute
    :param layer_attr: The extra layer attribute. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute | None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert isinstance(input, LayerOutput), 'prelu_layer accepts only one input.'

    if not param_attr:
        param_attr = ParamAttr(initial_mean=0.25, initial_std=0.0)
    else:
        assert isinstance(param_attr, ParameterAttribute)

    if num_channels is None:
        assert input.num_filters is not None, \
                'the input channel cannot be detected, please specify the num_channels parameter'
        num_channels = input.num_filters

    if channel_shared is not None:
        assert isinstance(channel_shared, bool)
        assert (input.height != 0 and input.width != 0), \
            'input height and widht must be setted'
        if channel_shared:
            partial_sum = input.height * input.width * num_channels
        else:
            partial_sum = input.height * input.width

    l = Layer(
        name=name,
        type=LayerType.PRELU,
        inputs=Input(input.name, **param_attr.attr),
        partial_sum=partial_sum,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        layer_type=LayerType.PRELU,
        parents=input,
        num_filters=num_channels,
        size=l.config.size)


@wrap_name_default()
@layer_support(ERROR_CLIPPING, DROPOUT)
@wrap_act_default(act=LinearActivation())
def gated_unit_layer(input,
                     size,
                     act=None,
                     name=None,
                     gate_attr=None,
                     gate_param_attr=None,
                     gate_bias_attr=True,
                     inproj_attr=None,
                     inproj_param_attr=None,
                     inproj_bias_attr=True,
                     layer_attr=None):
    """
    The gated unit layer implements a simple gating mechanism over the input.
    The input :math:`X` is first projected into a new space :math:`X'`, and
    it is also used to produce a gate weight :math:`\sigma`. Element-wise
    product between :math:`X'` and :math:`\sigma` is finally returned.

    Reference:
        `Language Modeling with Gated Convolutional Networks
        <https://arxiv.org/abs/1612.08083>`_

    .. math::
       y=\\text{act}(X \cdot W + b)\otimes \sigma(X \cdot V + c)

    The example usage is:

    .. code-block:: python
        gated_unit = gated_unit_layer(size=128, input=input_layer))

    :param input: The input of this layer.
    :type input: LayerOutput
    :param size: The dimension of this layer's output.
    :type size: int
    :param act: Activation type of the projection. LinearActivation is the default
                activation.
    :type act: BaseActivation
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param gate_attr: The extra layer attribute of the gate. See ExtraLayerAttribute for
                      details.
    :type gate_attr: ExtraLayerAttribute | None
    :param gate_param_attr: The parameter attribute of the gate. See ParameterAttribute
                            for details.
    :type gate_param_attr: ParameterAttribute
    :param gate_bias_attr: The bias attribute of the gate. If this parameter is set to False or
                           an object whose type is not ParameterAttribute, no bias is defined.
                           If this parameter is set to True, the bias is initialized to zero.
    :type gate_bias_attr: ParameterAttribute | bool | None | Any
    :param inproj_attr: Extra layer attributes of the projection. See ExtraLayerAttribute for
                        details.
    :type inproj_attr: ExtraLayerAttribute | None
    :param inproj_param_attr: The parameter attribute of the projection. See ParameterAttribute
                              for details.
    :type inproj_param_attr: ParameterAttribute
    :param inproj_bias_attr: The bias attribute of the projection. If this parameter is set to False
                             or an object whose type is not ParameterAttribute, no bias is defined.
                             If this parameter is set to True, the bias is initialized to zero.
    :type inproj_bias_attr: ParameterAttribute | bool | None | Any
    :param layer_attr: Extra layer attribute of the product. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute | None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert isinstance(
        input, LayerOutput), 'The gated linear unit accepts only one input.'

    input_proj = fc_layer(
        input=input,
        name="%s_input_proj" % name,
        size=size,
        act=act,
        layer_attr=inproj_attr,
        param_attr=inproj_param_attr,
        bias_attr=inproj_bias_attr)

    gate = fc_layer(
        size=size,
        name="%s_gate" % name,
        act=SigmoidActivation(),
        input=input,
        layer_attr=gate_attr,
        param_attr=gate_param_attr,
        bias_attr=gate_bias_attr)
    return mixed_layer(
        name="%s_gated_act" % name,
        input=dotmul_operator(input_proj, gate),
        layer_attr=layer_attr)


@layer_support()
@wrap_name_default('switch_order')
def switch_order_layer(input,
                       name=None,
                       reshape_axis=None,
                       act=None,
                       layer_attr=None):
    """
    This layer switch dimension order of image input.
    From order "batchSize, channels, height, width"
    to order "batchSize, height, width, channels".

    The example usage is:

    .. code-block:: python
       reshape_axis = 3
       switch = switch_order(input=layer, name='switch', reshape_axis=reshape_axis)
       reshape = {'height':[ 0, 1, 2], 'width':[3]}

    :param input: The input of this layer.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param reshape_axis: Specify the axises of 'height'. Its value should be positive and less than 4.
    :type reshape_axis: int
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput)
    assert reshape_axis != None and (reshape_axis > 0 and reshape_axis < 4)
    height = [ele for ele in xrange(reshape_axis)]
    width = [ele for ele in range(reshape_axis, 4)]
    reshape = {'height': height, 'width': width}

    l = Layer(
        name=name,
        inputs=input.name,
        reshape=reshape,
        type=LayerType.SWITCH_ORDER_LAYER,
        active_type=act.name,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        layer_type=LayerType.SWITCH_ORDER_LAYER,
        activation=act,
        parents=input,
        size=l.config.size)


@wrap_name_default()
@layer_support()
def crop_layer(input, offset, axis=2, shape=None, name=None, layer_attr=None):
    """
    This layer crops images according to the offset and shape. Users can set
    the crop shape through the argument 'shape' explicitly or by specifying a
    reference input layer.

    The example usage is:

    .. code-block:: python
    crop = crop_layer(input=[image_input, reference_input], axis=2, offset=[2, 3])

    :param input: The input of this layer. If two inputs are given, the second one
                  will be regarded as the reference.
                  And the input must be 4-dims and in NCHW order.
    :type input: LayerOutput | Sequence
    :param offset: The crop offset.
    :type offset: Sequence
    :param axis: The start axis to be cropped. For image input layer:
        - 0: batch size
        - 1: channels
        - 2: height
        - 3: width
    :type axis: int
    :param shape: The shape to be cropped to. Default is None.
    :type shape: Sequence | None
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if isinstance(input, LayerOutput):
        input = [input]
    else:
        assert isinstance(input, collections.Sequence)
    l = Layer(
        inputs=[x.name for x in input],
        axis=axis,
        offset=offset,
        shape=shape,
        name=name,
        type=LayerType.CROP_LAYER,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name=name,
        layer_type=LayerType.CROP_LAYER,
        parents=input,
        size=l.config.size)


@wrap_name_default()
@layer_support()
def sub_nested_seq_layer(input, selected_indices, name=None):
    """
    The sub_nested_seq_layer accepts two inputs: the first one is a nested
    sequence; the second one is a set of selceted indices in the nested sequence.

    Then sub_nest_seq_layer trims the first nested sequence input according
    to the selected indices to form a new output. This layer is useful in
    beam training.

    The example usage is:

    .. code-block:: python

        sub_nest_seq = sub_nested_seq_layer(input=data, selected_indices=selected_ids)


    :param input: The input of this layer. It is a nested sequence.
    :type input: LayerOutput
    :param selected_indices: A set of sequence indices in the nested sequence.
    :type input: LayerOutput
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert isinstance(input, LayerOutput), (
        'The first input of '
        'sub_nested_seq_layer must be a Paddle layer.')
    assert isinstance(selected_indices, LayerOutput), (
        'The second input of '
        'sub_nested_seq_layer must be a Paddle layer.')

    l = Layer(
        inputs=input.name,
        selected_indices=selected_indices.name,
        name=name,
        type=LayerType.SUB_NESTED_SEQ)
    return LayerOutput(
        name=name,
        layer_type=LayerType.SUB_NESTED_SEQ,
        parents=input,
        size=l.config.size)


@wrap_name_default("clip")
def clip_layer(input, min, max, name=None):
    """
    A layer for clipping the input value by the threshold.

    .. math::

        out[i] = \min (\max (in[i],p_{1} ),p_{2} )

    .. code-block:: python

        clip = clip_layer(input=input_layer, min=-10, max=10)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput.
    :param min: The lower threshold for clipping.
    :type min: float
    :param max: The upper threshold for clipping.
    :type max: float
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    Layer(
        name=name,
        type=LayerType.CLIP_LAYER,
        inputs=[input.name],
        min=min,
        max=max)
    return LayerOutput(
        name, LayerType.CLIP_LAYER, parents=[input], size=input.size)


@wrap_name_default()
def seq_slice_layer(input, starts, ends, name=None):
    """
    seq_slice_layer will return one or several sub-sequences from the
    input sequence layer given start and end indices.

        - If only start indices are given, and end indices are set to None,
          this layer slices the input sequence from the given start indices
          to its end.
        - If only end indices are given, and start indices are set to None,
          this layer slices the input sequence from its beginning to the
          given end indices.
        - If start and end indices are both given, they should have the same
          number of elements.

    If start or end indices contains more than one elements, the input sequence
    will be sliced for multiple times.


    .. code-block:: python

        seq_silce = seq_slice_layer(input=input_seq,
                                    starts=start_pos, ends=end_pos)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer, which should be a sequence.
    :type input: LayerOutput
    :param starts: The start indices to slice the input sequence.
    :type starts: LayerOutput | None
    :param ends: The end indices to slice the input sequence.
    :type ends: LayerOutput | None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert isinstance(input, LayerOutput), (
        'The first input of seq_slice layer must be a PaddlePaddle layer.')

    if starts is not None:
        assert isinstance(starts, LayerOutput), (
            'The start indices for seq_slice layer '
            'must be a PaddlePaddle layer.')
    if ends is not None:
        assert isinstance(ends, LayerOutput), (
            'The end indices for seq_slice layer must be a PaddlePaddle layer.')
    assert starts is not None or ends is not None, (
        'start and end indices '
        'cannot be set to None at the same time, at least one of '
        'them should be given.')
    if starts is not None and ends is not None:
        assert starts.size == ends.size, (
            'If start and end indices are both given to seq_slice_layer, '
            'they should have the same width.')

    Layer(
        name=name,
        type=LayerType.SEQ_SLICE,
        inputs=input.name,
        starts=starts.name if starts is not None else None,
        ends=ends.name if ends is not None else None)
    return LayerOutput(
        name, LayerType.SEQ_SLICE, parents=[input], size=input.size)


@wrap_name_default()
@layer_support()
def kmax_seq_score_layer(input, name=None, beam_size=1):
    """
    This layer accepts one input which is scores over a sequence or a nested
    sequence, and returns indices of beam_size sequences with highest scores.

    .. code-block:: python

        kmax_indices = kmax_seq_score_layer(input=input_layer, beam_size)


    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer. It stores scores over a sequence or
                  a nested sequence and its size must be 1.
    :type input: LayerOutput
    :param beam_size: The indices of the sequences with top beam_size scores are returned.
    :type beam_size: int
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput), ("kmax_seq_score_layer "
                                            "accepts only one input.")
    assert input.size == 1, (
        "input of kmax_seq_score_layer is a score "
        "over a sequence or a nested sequence, so its width must be 1.")

    Layer(
        name=name,
        type=LayerType.KMAX_SEQ_SCORE,
        inputs=[input.name],
        beam_size=beam_size)

    return LayerOutput(
        name, LayerType.KMAX_SEQ_SCORE, parents=[input], size=input.size)


@wrap_name_default("conv3d")
@wrap_param_attr_default()
@wrap_bias_attr_default()
@wrap_act_default(act=ReluActivation())
@layer_support(DROPOUT)
def img_conv3d_layer(input,
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
                     trans=False,
                     layer_type=None):
    """

    The example usage is:

    ..  code-block:: python

        conv = img_conv3d_layer(input=data, filter_size=1,
                              num_channels=8,
                              num_filters=16, stride=1,
                              bias_attr=False,
                              act=ReluActivation())

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param filter_size: The dimensions of the filter kernel along three axises. If the parameter
                        is set to one integer, the three dimensions will be same.
    :type filter_size: int | tuple | list
    :param num_filters: The number of filters. It is as same as the output image channel.
    :type num_filters: int
    :param act: Activation type. ReluActivation is the default activation.
    :type act: BaseActivation
    :param groups: The number of the filter groups.
    :type groups: int
    :param stride: The strides of the convolution along three axises. If the parameter
                   is set to one integer, the three strides will be same.
    :type stride: int | tuple | list
    :param padding: The numbers of padding along three axises. If the parameter is set to
                    one integer, they will be same.
    :type padding: int | tuple | list
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :param num_channels: The number of input channels. If the parameter is not set or
                         set to None, its actual value will be automatically set to
                         the channels number of the input.
    :type num_channels: int
    :param param_attr: The parameter attribute of the convolution. See ParameterAttribute for
                       details.
    :type param_attr: ParameterAttribute
    :param shared_biases: Whether biases will be shared between filters or not.
    :type shared_biases: bool
    :param layer_attr: The extra layer attributes. See ExtraLayerAttribute for
                       details.
    :type layer_attr: ExtraLayerAttribute
    :param trans: True if it is a convTransLayer, False if it is a convLayer
    :type trans: bool
    :param layer_type: Specify the layer type. If the parameter is set, it must be "deconv3d"
                       when trans=True. If not set, it will be automatically set to "deconv3d"
                       when trans=True and "conv3d" when trans=False.
    :type layer_type: basestring
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    if num_channels is None:
        assert input.num_filters is not None
        num_channels = input.num_filters

    if isinstance(filter_size, collections.Sequence):
        assert len(filter_size) == 3
        filter_size, filter_size_y, filter_size_z = filter_size
    else:
        filter_size_y = filter_size
        filter_size_z = filter_size

    if isinstance(stride, collections.Sequence):
        assert len(stride) == 3
        stride, stride_y, stride_z = stride
    else:
        stride_y = stride
        stride_z = stride

    if isinstance(padding, collections.Sequence):
        assert len(padding) == 3
        padding, padding_y, padding_z = padding
    else:
        padding_y = padding
        padding_z = padding

    if param_attr.attr.get('initial_smart'):
        # special initial for conv layers.
        init_w = (2.0 / (filter_size**2 * num_channels))**0.5
        param_attr.attr["initial_mean"] = 0.0
        param_attr.attr["initial_std"] = init_w
        param_attr.attr["initial_strategy"] = 0
        param_attr.attr["initial_smart"] = False

    if layer_type:
        if trans:
            assert layer_type in ["deconv3d"]
        lt = layer_type
    else:
        lt = LayerType.DECONV3D_LAYER if trans else LayerType.CONV3D_LAYER

    l = Layer(
        name=name,
        inputs=Input(
            input.name,
            conv=Conv3D(
                filter_size=filter_size,
                padding=padding,
                stride=stride,
                channels=num_channels,
                groups=groups,
                filter_size_y=filter_size_y,
                padding_y=padding_y,
                stride_y=stride_y,
                filter_size_z=filter_size_z,
                padding_z=padding_z,
                stride_z=stride_z),
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


@wrap_name_default("scale_shift")
@wrap_param_attr_default()
@wrap_bias_attr_default()
def scale_shift_layer(input, name=None, param_attr=None, bias_attr=None):
    """
    A layer applies a linear transformation to each element in each row of
    the input matrix. For each element, the layer first re-scales it and then
    adds a bias to it.

    This layer is very like the SlopeInterceptLayer, except the scale and
    bias are trainable.

    .. math::

        y = w * x + b

    .. code-block:: python

        scale_shift = scale_shift_layer(input=input_layer, bias_attr=False)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer.
    :type input: LayerOutput
    :param param_attr: The parameter attribute of scaling. See ParameterAttribute for
                      details.
    :type param_attr: ParameterAttribute
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    Layer(
        name=name,
        type=LayerType.SCALE_SHIFT_LAYER,
        inputs=Input(input.name, **param_attr.attr),
        bias=ParamAttr.to_bias(bias_attr))
    return LayerOutput(
        name, LayerType.SCALE_SHIFT_LAYER, parents=[input], size=input.size)


@wrap_name_default("resize")
def resize_layer(input, size, name=None):
    """
    The resize layer resizes the input matrix with a shape of [Height, Width]
    into the output matrix with a shape of [Height x Width / size, size],
    where size is the parameter of this layer indicating the output dimension.

    :param input: The input of this layer.
    :type input: LayerOutput.
    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param size: The resized output dimension of this layer.
    :type size: int
    :return: A LayerOutput object.
    :rtype: LayerOutput
    """
    Layer(name=name, type=LayerType.RESIZE, inputs=Input(input.name), size=size)
    return LayerOutput(name, LayerType.RESIZE, parents=[input], size=input.size)


@wrap_act_default(act=LinearActivation())
@wrap_name_default('sub_seq')
def sub_seq_layer(input, offsets, sizes, act=None, bias_attr=None, name=None):
    """
    sub_seq_layer will return sub-sequences from the input sequences. For each
    sequence in the input sequence layer, sub_seq_layer will slice it by given
    offset and size. Please notice that, number of offset value and size value
    both are equal to the number of sequence in the input layer.

    .. code-block:: python

        sub_seq = sub_seq_layer(input=input_seq, offsets=offsets, sizes=sizes)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer, which should be sequence.
    :type input: LayerOutput
    :param offsets: The offset indices to slice the input sequence, which should
                    be sequence type.
    :type offsets: LayerOutput
    :param sizes: The sizes of the sub-sequences, which should be sequence type.
    :type sizes: LayerOutput
    :param act: Activation type, LinearActivation is the default activation.
    :type act: BaseActivation.
    :param bias_attr: The bias attribute. If the parameter is set to False or an object
                      whose type is not ParameterAttribute, no bias is defined. If the
                      parameter is set to True, the bias is initialized to zero.
    :type bias_attr: ParameterAttribute | None | bool | Any
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert isinstance(input, LayerOutput), (
        'The first input of sub_seq_layer layer must be a PaddlePaddle layer.')
    assert isinstance(offsets, LayerOutput), (
        'The offset indices for sub_seq_layer, '
        'must be a PaddlePaddle layer.')
    assert isinstance(sizes, LayerOutput), (
        'The sizes of sub-sequences, must be a PaddlePaddle layer.')

    Layer(
        name=name,
        type=LayerType.SUB_SEQ_LAYER,
        inputs=[input.name, offsets.name, sizes.name],
        active_type=act.name,
        bias=ParamAttr.to_bias(bias_attr))

    return LayerOutput(
        name,
        LayerType.SUB_SEQ_LAYER,
        parents=[input, offsets, sizes],
        size=input.size)


@wrap_name_default('scale_sub_region')
def scale_sub_region_layer(input, indices, value, name=None):
    """
    Given an image or feature map with CHW information, scale_sub_region_layer
    can be used to multiply a real value to values of a sub continuous region.
    You can provide start and end indices of CHW for each instance.
    Please notice that all start indices are counting from 1.
    The shape of indices should be [batch_size, 6] and the layout for each row
    is [C_Start, C_End, H_Start, H_End, W_Start, W_End].

    .. code-block:: python

        scale_sub_region = scale_sub_region_layer(input=input,
                                                  indices=indices,
                                                  value=value)

    :param name: The name of this layer. It is optional.
    :type name: basestring
    :param input: The input of this layer which should contains CHW information.
    :type input: LayerOutput
    :param indices: Start index and end index for C H W, the input value should
                    be a 2-D matrix with shape [batch_size, 6].
    :type indices: LayerOutput.
    :param value: value to multiply.
    :type value: float
    :return: LayerOutput object.
    :rtype: LayerOutput
    """

    assert isinstance(input, LayerOutput), (
        'The first input of scale_sub_region_layer, '
        'must be a PaddlePaddle layer.')
    assert isinstance(indices, LayerOutput), (
        'The start and end indices for CHW, must be a PaddlePaddle layer.')
    assert isinstance(value, float), (
        'The value to multiply, must be a real value.')

    Layer(
        name=name,
        type=LayerType.SCALE_SUB_REGION_LAYER,
        inputs=[input.name, indices.name],
        value=value)

    return LayerOutput(
        name,
        LayerType.SCALE_SUB_REGION_LAYER,
        parents=[input, indices],
        num_filters=input.num_filters,
        size=input.size)


@wrap_name_default()
@wrap_act_default(act=LinearActivation())
@wrap_param_attr_default()
@layer_support()
def factorization_machine(input,
                          factor_size,
                          act=None,
                          name=None,
                          param_attr=None,
                          layer_attr=None):
    """
    The Factorization Machine models pairwise feature interactions as inner
    product of the learned latent vectors corresponding to each input feature.
    The Factorization Machine can effectively capture feature interactions
    especially when the input is sparse.

    This implementation only consider the 2-order feature interactions using
    Factorization Machine with the formula:

    .. math::
        y = \sum_{i=1}^{n-1}\sum_{j=i+1}^n\langle v_i, v_j \\rangle x_i x_j

    Note:
        X is the input vector with size n. V is the factor matrix. Each row of V
        is the latent vector corresponding to each input dimesion. The size of
        each latent vector is k.

    For details of Factorization Machine, please refer to the paper:
    Factorization machines.

    .. code-block:: python
        first_order = paddle.layer.fc(input=input,
                                      size=1,
                                      act=paddle.activation.Linear())
        second_order = paddle.layer.factorization_machine(input=input,
                                                          factor_size=10)
        fm = paddle.layer.addto(input=[first_order, second_order],
                                act=paddle.activation.Linear(),
                                bias_attr=False)

    :param input: The input layer. Supported input types: all input data types
                  on CPU, and only dense input types on GPU.
    :type input: LayerOutput
    :param factor_size: The hyperparameter that defines the dimensionality of
                        the latent vector size.
    :type context_len: int
    :param act: Activation Type. Default is linear activation.
    :type act: BaseActivation
    :param param_attr: The parameter attribute. See ParameterAttribute for
                       details.
    :type param_attr: ParameterAttribute
    :param layer_attr: Extra Layer config.
    :type layer_attr: ExtraLayerAttribute|None
    :return: LayerOutput object.
    :rtype: LayerOutput
    """
    assert isinstance(input, LayerOutput)
    assert factor_size > 0, "the factor_size must be greater than 0."

    Layer(
        inputs=[Input(input.name, **param_attr.attr)],
        name=name,
        factor_size=factor_size,
        type=LayerType.FACTORIZATION_MACHINE,
        active_type=act.name,
        **ExtraLayerAttribute.to_kwargs(layer_attr))
    return LayerOutput(
        name, LayerType.FACTORIZATION_MACHINE, input, activation=act, size=1)
