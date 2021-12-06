// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/fluid/string/string_helper.h"
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/framework/fleet/ascend_wrapper.h"
#endif

std::set<std::string> gen_list = {
    "sigmoid",
    "matmul_v2",
    "reduce_sum",
    "elementwise_add",
    "rsqrt",
    "multihead_matmul",
    "addmm",
    "gru",
    "round",
    "push_dense",
    "rank_attention",
    "fused_embedding_fc_lstm",
    "where_index",
    "bicubic_interp",
    "arg_min",
    "tile",
    "bilinear_tensor_product",
    "ctc_align",
    "pow2_decay_with_linear_warmup",
    "marker",
    "split",
    "fc",
    "clear_float_status",
    "load",
    "elementwise_max",
    "adadelta",
    "check_finite_and_unscale",
    "sparse_momentum",
    "tan",
    "adam",
    "fsp",
    "where",
    "logical_xor",
    "multiclass_nms3",
    "one_hot_v2",
    "sequence_softmax",
    "affine_channel",
    "triangular_solve",
    "sequence_topk_avg_pooling",
    "space_to_depth",
    "reverse",
    "fused_embedding_eltwise_layernorm",
    "expand_v2",
    "lgamma",
    "solve",
    "deformable_psroi_pooling",
    "instance_norm",
    "decode_jpeg",
    "gather_nd",
    "reduce_prod",
    "matrix_rank",
    "asin",
    "lstmp",
    "iou_similarity",
    "huber_loss",
    "one_hot",
    "sequence_slice",
    "lookup_table",
    "softplus",
    "depthwise_conv2d",
    "fused_fc_elementwise_layernorm",
    "sigmoid_cross_entropy_with_logits",
    "exp",
    "scatter",
    "equal_all",
    "searchsorted",
    "fusion_squared_mat_sub",
    "unique",
    "log",
    "conv_shift",
    "smooth_l1_loss",
    "linear_interp_v2",
    "momentum",
    "temporal_shift",
    "nce",
    "mv",
    "proximal_gd",
    "memcpy_h2d",
    "add_position_encoding",
    "cosh",
    "hash",
    "grad_add",
    "sign",
    "prelu",
    "linspace",
    "fill_diagonal",
    "logsigmoid",
    "load_combine",
    "fetch_v2",
    "randperm",
    "sequence_scatter",
    "partial_sum",
    "relu6",
    "conv3d",
    "lstm_unit",
    "not_equal",
    "transpose2",
    "uniform_random_batch_size_like",
    "unfold",
    "lrn",
    "softmax_with_cross_entropy",
    "isfinite_v2",
    "bernoulli",
    "max_pool3d_with_index",
    "gaussian_random",
    "flatten2",
    "matmul",
    "cvm",
    "adamax",
    "masked_select",
    "range",
    "bitwise_not",
    "trace",
    "multinomial",
    "modified_huber_loss",
    "roll",
    "squared_l2_distance",
    "conv3d_transpose",
    "share_data",
    "fake_quantize_abs_max",
    "unique_with_counts",
    "fill",
    "concat",
    "fill_zeros_like",
    "hierarchical_sigmoid",
    "isinf_v2",
    "squeeze",
    "multiclass_nms2",
    "bpr_loss",
    "fft_c2c",
    "bicubic_interp_v2",
    "reshape",
    "coalesce_tensor",
    "roi_align",
    "reshape2",
    "reduce_any",
    "unstack",
    "scatter_nd_add",
    "sequence_reshape",
    "bilateral_slice",
    "fill_any_like",
    "empty",
    "pad_constant_like",
    "pool2d",
    "size",
    "imag",
    "eigh",
    "stack",
    "dgc_momentum",
    "lamb",
    "generate_proposals_v2",
    "bitwise_or",
    "gru_unit",
    "fake_channel_wise_quantize_dequantize_abs_max",
    "sampling_id",
    "unsqueeze2",
    "average_accumulates",
    "sequence_enumerate",
    "fusion_seqconv_eltadd_relu",
    "bce_loss",
    "generate_proposal_labels",
    "im2sequence",
    "isinf",
    "adagrad",
    "linear_chain_crf",
    "retinanet_target_assign",
    "fusion_group",
    "teacher_student_sigmoid_loss",
    "random_crop",
    "lookup_table_v2",
    "detection_map",
    "l1_norm",
    "sqrt",
    "fused_elemwise_activation",
    "slogdeterminant",
    "share_buffer",
    "bitwise_and",
    "diag_embed",
    "unbind",
    "dropout",
    "moving_average_abs_max_scale",
    "beam_search",
    "log_loss",
    "greater_than",
    "kron",
    "sigmoid_focal_loss",
    "rmsprop",
    "conv2d",
    "uniform_random_inplace",
    "maxout",
    "linear_interp",
    "auc",
    "logical_or",
    "batch_norm",
    "acos",
    "unpool",
    "cumprod",
    "sample_logits",
    "crop_tensor",
    "fill_constant",
    "deformable_conv",
    "generate_mask_labels",
    "locality_aware_nms",
    "expand_as",
    "matrix_power",
    "greater_equal",
    "generate_proposals",
    "bilinear_interp",
    "inplace_abn",
    "softshrink",
    "mul",
    "data_norm",
    "get_tensor_from_selected_rows",
    "spp",
    "floor",
    "gelu",
    "retinanet_detection_output",
    "push_dense",
    "silu",
    "sequence_erase",
    "real",
    "nearest_interp_v2",
    "dgc_clip_by_norm",
    "squeeze2",
    "strided_slice",
    "conj",
    "precision_recall",
    "save",
    "fusion_seqexpand_concat_fc",
    "fake_quantize_range_abs_max",
    "depthwise_conv2d_transpose",
    "positive_negative_pair",
    "square",
    "var_conv_2d",
    "log1p",
    "fused_softmax_mask_upper_triangle",
    "clip_by_norm",
    "atan2",
    "box_decoder_and_assign",
    "fft_r2c",
    "roi_pool",
    "overlap_add",
    "fill_constant_batch_size_like",
    "fill_any",
    "dequantize_log",
    "max_pool2d_with_index",
    "pad3d",
    "norm",
    "viterbi_decode",
    "mish",
    "box_coder",
    "flatten",
    "elementwise_mod",
    "margin_cross_entropy",
    "logical_and",
    "pow",
    "stanh",
    "label_smooth",
    "merged_momentum",
    "ascend_trigger",
    "fused_feedforward",
    "rpn_target_assign",
    "roi_perspective_transform",
    "expand",
    "prroi_pool",
    "pool3d",
    "memcpy",
    "distribute_fpn_proposals",
    "frame",
    "bincount",
    "shape",
    "group_norm",
    "resnet_unit",
    "sequence_expand_as",
    "cos_sim",
    "eigvals",
    "save_combine",
    "class_center_sample",
    "read_file",
    "isfinite",
    "arg_max",
    "equal",
    "fake_dequantize_max_abs",
    "qr",
    "anchor_generator",
    "layer_norm",
    "merge_selected_rows",
    "less_equal",
    "rnn",
    "fusion_lstm",
    "lars_momentum",
    "hard_sigmoid",
    "isnan",
    "elementwise_floordiv",
    "correlation",
    "histogram",
    "gather_tree",
    "segment_pool",
    "sync_batch_norm",
    "fusion_repeated_fc_relu",
    "nop",
    "expand_as_v2",
    "filter_by_instag",
    "nll_loss",
    "dot",
    "scale",
    "ncclBcast",
    "shuffle_batch",
    "ncclReduce",
    "diag",
    "multiplex",
    "leaky_relu",
    "allclose",
    "adamw",
    "elementwise_pow",
    "prior_box",
    "p_norm",
    "unique_consecutive",
    "lod_reset",
    "pad",
    "sequence_conv",
    "log10",
    "set_value",
    "bitwise_xor",
    "center_loss",
    "randint",
    "attention_lstm",
    "uniform_random",
    "slice",
    "meshgrid",
    "hard_swish",
    "sin",
    "mean_iou",
    "pad2d",
    "inverse",
    "spectral_norm",
    "shuffle_channel",
    "psroi_pool",
    "seed",
    "ceil",
    "eig",
    "reduce_min",
    "cos",
    "ncclAllReduce",
    "cudnn_lstm",
    "digamma",
    "assign_value",
    "increment",
    "tdm_sampler",
    "fused_softmax_mask",
    "sequence_reverse",
    "eigvalsh",
    "diagonal",
    "trunc",
    "log2",
    "tanh",
    "yolov3_loss",
    "graph_send_recv",
    "accuracy",
    "atan",
    "less_than",
    "unsqueeze",
    "crf_decoding",
    "log_softmax",
    "ftrl",
    "matrix_nms",
    "top_k_v2",
    "cast",
    "tanh_shrink",
    "hard_shrink",
    "multiclass_nms",
    "fusion_transpose_flatten_concat",
    "sequence_unpad",
    "fused_elemwise_add_activation",
    "frobenius_norm",
    "crop",
    "cross_entropy2",
    "skip_layernorm",
    "tdm_child",
    "fused_embedding_seq_pool",
    "erf",
    "conv2d_inception_fusion",
    "trilinear_interp",
    "logsumexp",
    "fusion_seqpool_concat",
    "alloc_float_status",
    "sequence_concat",
    "fusion_seqpool_cvm_concat",
    "similarity_focus",
    "argsort",
    "sequence_expand",
    "sgd",
    "fused_bn_add_activation",
    "bilinear_interp_v2",
    "clip",
    "deformable_conv_v1",
    "hinge_loss",
    "determinant",
    "conv2d_transpose",
    "memcpy_d2h",
    "softsign",
    "fake_quantize_dequantize_abs_max",
    "broadcast_tensors",
    "grid_sampler",
    "fft_c2r",
    "pyramid_hash",
    "fake_quantize_dequantize_moving_average_abs_max",
    "multi_dot",
    "sequence_pool",
    "transpose",
    "top_k",
    "dist",
    "affine_grid",
    "gaussian_random_batch_size_like",
    "fake_channel_wise_dequantize_max_abs",
    "reciprocal",
    "sequence_mask",
    "fill_diagonal_tensor",
    "abs",
    "partial_concat",
    "elu",
    "index_select",
    "row_conv",
    "cross",
    "elementwise_mul",
    "decayed_adagrad",
    "bipartite_match",
    "run_program",
    "fake_quantize_moving_average_abs_max",
    "mine_hard_examples",
    "target_assign",
    "lstm",
    "truncated_gaussian_random",
    "match_matrix_tensor",
    "elementwise_div",
    "kldiv_loss",
    "cumsum",
    "sum",
    "proximal_adagrad",
    "update_loss_scaling",
    "shard_index",
    "selu",
    "mean",
    "gumbel_softmax",
    "sequence_pad",
    "tree_conv",
    "assign",
    "flatten_contiguous_range",
    "tril_triu",
    "brelu",
    "celu",
    "reduce_mean",
    "sinh",
    "rank_loss",
    "reduce_max",
    "fusion_gru",
    "fill_zeros_like2",
    "expm1",
    "squared_l2_norm",
    "elementwise_sub",
    "margin_rank_loss",
    "faster_tokenizer",
    "relu",
    "is_empty",
    "reduce_all",
    "edit_distance",
    "bmm",
    "yolo_box",
    "soft_relu",
    "density_prior_box",
    "eye",
    "swish",
    "cross_entropy",
    "dpsgd",
    "cholesky",
    "batch_fc",
    "nearest_interp",
    "gather",
    "trilinear_interp_v2",
    "box_clip",
    "isnan_v2",
    "softmax",
    "conv2d_fusion",
    "fused_batch_norm_act",
    "get_float_status",
    "index_sample",
    "elementwise_min",
    "logical_not",
    "collect_fpn_proposals",
    "pixel_shuffle",
    "thresholded_relu",
    "polygon_box_transform",
    "lookup_table_dequant",
    "warpctc",
    "fake_channel_wise_quantize_abs_max",
    "dequantize_abs_max",
    "svd",
    "flip"};

// NOTE(zhiqiu): Commonly, the inputs in auto-generated OP function are
// determined by the OP`s proto automatically, i.e., all the inputs registered
// in OpMaker.
// However, some OPs have dispensable inputs, which means the input can
// be none for some conditions. It is discovered that most dispensable inputs
// is not used in imperative mode, so we drop those inputs when generating OP
// functions. While, for very few OPs, the dispensable inputs are used, we
// need to manually specify them in this map.
std::map<std::string, std::set<std::string>> op_ins_map = {
    {"layer_norm", {"X", "Scale", "Bias"}},
    {"fused_attention",
     {"X", "LnScale", "LnBias", "QKVW", "QKVBias", "SrcMask", "OutLinearW",
      "OutLinearBias", "Ln2Scale", "Ln2Bias"}},
    {"instance_norm", {"X", "Scale", "Bias"}},
    {"gru_unit", {"Input", "HiddenPrev", "Weight", "Bias"}},
    {"label_smooth", {"X", "PriorDist"}},
    {"assign", {"X"}},
    {"reshape2", {"X", "Shape"}},
    {"expand", {"X", "ExpandTimes"}},
    {"slice", {"Input", "StartsTensor", "EndsTensor"}},
    {"fake_quantize_dequantize_moving_average_abs_max",
     {"X", "InScale", "InAccum", "InState"}},
    {"nll_loss", {"X", "Label", "Weight"}},
    {"bilinear_tensor_product", {"X", "Y", "Weight", "Bias"}},
    {"gather", {"X", "Index", "Axis"}},
    {"roi_pool", {"X", "ROIs", "RoisNum"}},
    {"roi_align", {"X", "ROIs", "RoisNum"}},
    {"psroi_pool", {"X", "ROIs", "RoisNum"}},
    {"collect_fpn_proposals",
     {"MultiLevelRois", "MultiLevelScores", "MultiLevelRoIsNum"}},
    {"distribute_fpn_proposals", {"FpnRois", "RoisNum"}},
    {"warpctc", {"Logits", "Label", "LogitsLength", "LabelLength"}},
    {"hierarchical_sigmoid",
     {"X", "W", "Label", "PathTable", "PathCode", "Bias"}},
    {"moving_average_abs_max_scale", {"X", "InAccum", "InState"}},
    {"multiclass_nms3", {"BBoxes", "Scores", "RoisNum"}},
    {"box_coder", {"PriorBox", "PriorBoxVar", "TargetBox"}},
    {"momentum", {"Param", "Grad", "Velocity", "LearningRate", "MasterParam"}},
    {"sparse_momentum", {"Param", "Grad", "Velocity", "Index", "LearningRate"}},
    {"rnn", {"Input", "PreState", "WeightList", "SequenceLength"}},
    {"run_program", {"X", "Params"}},
    {"faster_tokenizer", {"Text", "Vocab", "TextPair"}},
    {"matrix_rank", {"X", "TolTensor"}},
    {"adam",
     {"Param", "Grad", "LearningRate", "Moment1", "Moment2", "Beta1Pow",
      "Beta2Pow", "MasterParam"}},
    {"adamw",
     {"Param", "Grad", "LearningRate", "Moment1", "Moment2", "Beta1Pow",
      "Beta2Pow", "MasterParam"}},
};

// NOTE(zhiqiu): Like op_ins_map.
// Commonly, the outputs in auto-generated OP function are determined by the
// OP`s proto automatically, i.e., all the outputs registered in OpMaker.
// However, some OPs have dispensable outputs, which means the output can
// be none for some conditions. It is discovered that most dispensable outputs
// is not used in imperative mode, so we drop those outputs when generating OP
// functions. While, for very few OPs, the dispensable outputs are used, we
// need to manually specify them in this map.
std::map<std::string, std::set<std::string>> op_outs_map = {
    {"fake_quantize_dequantize_moving_average_abs_max",
     {"Out", "OutScale", "OutAccum", "OutState"}},
    {"batch_norm",
     {"Y", "MeanOut", "VarianceOut", "SavedMean", "SavedVariance",
      "ReserveSpace"}},
    {"fused_attention",
     {"LnMean", "LnVariance", "LnOut", "QKVOut", "QKVBiasOut", "TransposeOut2",
      "QKOut", "QKTVOut", "SoftmaxOut", "AttnDropoutMaskOut", "AttnDropoutOut",
      "SrcMaskOut", "FMHAOut", "OutLinearOut", "DropoutMaskOut", "Ln2Mean",
      "Ln2Variance", "BiasDropoutResidualOut", "Y"}},
    {"sync_batch_norm",
     {"Y", "MeanOut", "VarianceOut", "SavedMean", "SavedVariance",
      "ReserveSpace"}},
    {"unique", {"Out", "Index", "Indices", "Counts"}},
    {"unique_consecutive", {"Out", "Index", "Counts"}},
    {"generate_proposals", {"RpnRois", "RpnRoiProbs", "RpnRoisNum"}},
    {"collect_fpn_proposals", {"FpnRois", "RoisNum"}},
    {"matrix_nms", {"Out", "Index", "RoisNum"}},
    {"distribute_fpn_proposals",
     {"MultiFpnRois", "RestoreIndex", "MultiLevelRoIsNum"}},
    {"moving_average_abs_max_scale",
     {"Out", "OutScale", "OutAccum", "OutState"}},
    {"multiclass_nms3", {"Out", "NmsRoisNum"}},
    {"generate_proposals_v2", {"RpnRois", "RpnRoiProbs", "RpnRoisNum"}},
    {"momentum", {"ParamOut", "VelocityOut", "MasterParamOut"}},
    {"sparse_momentum", {"ParamOut", "VelocityOut"}},
    {"rnn", {"DropoutState", "Reserve", "Out", "State"}},
    {"lamb",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut"}},
    {"run_program", {"DOut"}},
    {"adam",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut",
      "MasterParamOut"}},
    {"adamw",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut",
      "MasterParamOut"}},
};

// NOTE(zhiqiu): Commonly, the outputs in auto-generated OP function are
// generated in C++ automatically.
// However, some OPs need to pass the outputs from Python instead of generating
// them in C++. There are mainly 2 reasons for that,
// (1) Optimizer OPs need to update the input param in-place, like sgd.
//     So they need to pass the output which is same as input param.
// (2) Very few python APIs has out in their arguments, like fill_constant.
//     So they need to pass the python output to C++.
//     Actually, this is not a good design, since it may break the SSA graph,
//     especially in declarative mode.
// For those OPs, we need to manually specify the outs need to pass in this map.
std::map<std::string, std::set<std::string>> op_passing_outs_map = {
    {"sgd", {"ParamOut"}},
    {"adam",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut",
      "MasterParamOut"}},
    {"adamw",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut",
      "MasterParamOut"}},
    {"average_accumulates",
     {"out_sum_1", "out_sum_2", "out_sum_3", "out_num_accumulates",
      "out_old_num_accumulates", "out_num_updates"}},
    {"momentum", {"ParamOut", "VelocityOut", "MasterParamOut"}},
    {"sparse_momentum", {"ParamOut", "VelocityOut"}},
    {"batch_norm", {"MeanOut", "VarianceOut"}},
    {"sync_batch_norm", {"MeanOut", "VarianceOut"}},
    {"accuracy", {"Correct", "Total"}},
    {"fill_constant", {"Out"}},
    {"recv_v2", {"Out"}},
    {"partial_recv", {"Out"}},
    {"matmul", {"Out"}},
    {"c_broadcast", {"Out"}},
    {"c_sync_calc_stream", {"Out"}},
    {"c_sync_comm_stream", {"Out"}},
    {"c_reduce_sum", {"Out"}},
    {"c_reduce_max", {"Out"}},
    {"c_reduce_min", {"Out"}},
    {"c_reduce_prod", {"Out"}},
    {"c_reduce", {"Out"}},
    {"c_scatter", {"Out"}},
    {"barrier", {"Out"}},
    {"fake_quantize_dequantize_moving_average_abs_max",
     {"Out", "OutScale", "OutAccum", "OutState"}},
    {"fake_quantize_dequantize_abs_max", {"Out", "OutScale"}},
    {"fake_channel_wise_quantize_dequantize_abs_max", {"Out", "OutScale"}},
    {"check_finite_and_unscale", {"Out", "FoundInfinite"}},
    {"update_loss_scaling",
     {"Out", "LossScaling", "OutGoodSteps", "OutBadSteps"}},
    {"moving_average_abs_max_scale",
     {"Out", "OutScale", "OutAccum", "OutState"}},
    {"lamb",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut"}},
    {"rnn", {"DropoutState"}},
    {"run_program", {"Out", "DOut", "OutScope"}},
    {"clear_float_status", {"FloatStatusOut"}},
    {"get_float_status", {"FloatStatusOut"}},
};

// NOTE(pangyoki): Tensor View Strategy.
// In this case, a new output varbase will be created, and this varbase will
// reuse the input varbase's allocation.
// It's a map. The key of outer map is the view op name, the value is
// a pair which implies the mapping relationship between the input and
// output varbase.
std::map<std::string, std::pair<std::string, std::string>> view_op_map = {
    {"squeeze2", {"X", "Out"}},  // "X" -> "Out"
    {"unsqueeze2", {"X", "Out"}},
    {"reshape2", {"X", "Out"}},
    {"flatten_contiguous_range", {"X", "Out"}},
};

// clang-format off
const char* OUT_INITIALIZER_TEMPLATE =
    R"({"%s", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}})";
const char* OUT_DUPLICABLE_INITIALIZER_TEMPLATE = R"({"%s", ConstructDuplicableOutput(%s)})";

const char* INPUT_INITIALIZER_TEMPLATE = R"({"%s", {%s}})";
const char* INPUT_LIST_INITIALIZER_TEMPLATE = R"({"%s", %s})";

const char* INPUT_INITIALIZER_TEMPLATE_WITH_NULL = R"(
    if (%s != nullptr) {
      ins["%s"] = {%s};
    }
)";

const char* INPUT_INITIALIZER_TEMPLATE_WITH_NULL_LIST = R"(
    if (%s.size() != 0) {
      ins["%s"] = %s;
    }
)";

const char* OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL = R"(
    outs["%s"] = {%s};
)";

const char* OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL_LIST = R"(
    outs["%s"] = %s;
)";
// if inputs is list, no need {}
const char* ARG_OUT_NUM = R"(%sNum)";
const char* ARG_OUT_NUM_TYPE = R"(size_t )";

const char* IN_VAR_TYPE = R"(py::handle)";
const char* IN_VAR_LIST_TYPE = R"(py::handle)";

const char* OUT_VAR_TYPE = R"(std::shared_ptr<imperative::VarBase>)";
const char* OUT_VAR_LIST_TYPE = R"(std::vector<std::shared_ptr<imperative::VarBase>>)";

const char* CAST_VAR_TEMPLATE = R"(
    auto %s = GetEagerTensorFromArgs("%s", "%s", args, %d, %s);)";

const char* CAST_VAR_LIST_TEMPLATE = R"(
    auto %s = GetEagerTensorListFromArgs("%s", "%s", args, %d, %s);)";

const char* CAST_SIZE_T_TEMPLATE = R"(
    auto %s = GetUnsignedLongFromArgs("%s", "%s", args, %d, %s);)";

const char* ARG_TEMPLATE = R"(const %s& %s)";

const char* RETURN_TUPLE_TYPE = R"(std::tuple<%s>)";
const char* RETURN_TUPLE_TEMPLATE = R"(std::make_tuple(%s))";
const char* RETURN_LIST_TEMPLATE = R"(outs["%s"])";
const char* RETURN_TEMPLATE = R"(outs["%s"][0])";

const char* FUNCTION_ARGS = R"(%s, const py::args& args)";
const char* FUNCTION_ARGS_NO_INPUT = R"(const py::args& args)";

const char* HANDLE_VIEW_BETWEEN_INPUT_AND_OUTPUT = R"(
    if (ins.count("%s") && outs.count("%s")) {
      HandleViewBetweenInputAndOutput(ins["%s"][0], outs["%s"][0]);
    })";

const char* OP_FUNCTION_TEMPLATE =
R"(
static PyObject * %s(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    %s
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("%s", args, %d, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    %s
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    %s
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
})";

const char* PYBIND_ITEM_TEMPLATE = R"(  {"%s", (PyCFunction)(void(*)(void))%s, METH_VARARGS | METH_KEYWORDS, "C++ interface function for %s in dygraph."},)";

// clang-format on
static inline bool FindInsMap(const std::string& op_type,
                              const std::string& in_name) {
  return op_ins_map[op_type].count(in_name);
}

static inline bool FindOutsMap(const std::string& op_type,
                               const std::string& out_name) {
  return op_outs_map[op_type].count(out_name);
}

static inline bool FindPassingOutsMap(const std::string& op_type,
                                      const std::string& out_name) {
  return op_passing_outs_map[op_type].count(out_name);
}

static inline bool FindViewOpMap(const std::string& op_type) {
  return view_op_map.count(op_type);
}

static inline std::string TempName(const std::string& name) {
  return name + '_';
}

std::string GenerateOpFunctionsBody(
    const paddle::framework::proto::OpProto* op_proto, std::string func_name,
    bool use_inplace_strategy = false,
    std::map<std::string, std::string> inplace_map = {}) {
  auto& op_type = op_proto->type();
  std::string input_args = "";
  std::string call_api_str = "auto out = " + op_type + "_dygraph_function(";
  std::string ins_initializer_with_null = "";
  std::string py_arg = "";
  int arg_idx = 0;
  int input_args_num = 0;
  std::string ins_cast_str = "";
  std::string view_strategy_str = "";
  for (auto& input : op_proto->inputs()) {
    auto& in_name = input.name();
    // skip those dispensable inputs, like ResidualData in conv2d
    if (input.dispensable() && !FindInsMap(op_type, in_name)) {
      continue;
    }
    const auto in_type = input.duplicable() ? IN_VAR_LIST_TYPE : IN_VAR_TYPE;
    auto input_arg =
        paddle::string::Sprintf(ARG_TEMPLATE, in_type, TempName(in_name));
    input_args += input_arg;
    input_args += ",";
    input_args_num++;
    const auto in_cast_type =
        input.duplicable() ? CAST_VAR_LIST_TEMPLATE : CAST_VAR_TEMPLATE;
    auto dispensable = input.dispensable() ? "true" : "false";
    ins_cast_str += paddle::string::Sprintf(in_cast_type, in_name, op_type,
                                            in_name, arg_idx++, dispensable);

    call_api_str += in_name + ", ";
  }

  if (!input_args.empty() && input_args.back() == ',') {
    input_args.pop_back();
  }

  // Generate outs initializer
  std::string outs_initializer = "{";
  std::string outs_initializer_with_null = "";
  std::string return_str = "";

  int outs_num = 0;
  for (auto& output : op_proto->outputs()) {
    auto& out_name = output.name();

    // skip those dispensable oututs
    if (output.dispensable() && !FindOutsMap(op_type, out_name)) {
      continue;
    }
    const auto out_type =
        output.duplicable() ? OUT_VAR_LIST_TYPE : OUT_VAR_TYPE;

    if (FindPassingOutsMap(op_type, out_name)) {
      if (input_args != "") {
        input_args += ",";
      }
      input_args += out_type;
      input_args += out_name;
      input_args_num++;

      if (output.dispensable()) {
        const auto out_template =
            output.duplicable() ? OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL_LIST
                                : OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL;
        outs_initializer_with_null +=
            paddle::string::Sprintf(out_template, out_name, out_name);
      } else {
        const auto out_template = output.duplicable()
                                      ? INPUT_LIST_INITIALIZER_TEMPLATE
                                      : INPUT_INITIALIZER_TEMPLATE;
        outs_initializer +=
            paddle::string::Sprintf(out_template, out_name, out_name);
        outs_initializer += ",";
      }

      const auto in_cast_type =
          output.duplicable() ? CAST_VAR_LIST_TEMPLATE : CAST_VAR_TEMPLATE;
      auto dispensable = output.dispensable() ? "true" : "false";
      ins_cast_str += paddle::string::Sprintf(in_cast_type, out_name, op_type,
                                              out_name, arg_idx++, dispensable);
    } else {
      // There are few Operators that have duplicable output, like `Out` in
      // split op. We need to specify the number of variables for the
      // duplicable output, as the argument OutNum;
      if (output.duplicable()) {
        if (input_args != "") {
          input_args += ",";
        }
        auto out_num_str = paddle::string::Sprintf(ARG_OUT_NUM, out_name);
        input_args += ARG_OUT_NUM_TYPE;
        input_args += out_num_str;
        input_args_num++;
        outs_initializer += paddle::string::Sprintf(
            OUT_DUPLICABLE_INITIALIZER_TEMPLATE, out_name, out_num_str);

        auto dispensable = output.dispensable() ? "true" : "false";
        ins_cast_str +=
            paddle::string::Sprintf(CAST_SIZE_T_TEMPLATE, out_num_str, op_type,
                                    out_num_str, arg_idx++, dispensable);
        call_api_str += out_num_str + ", ";
      } else {
        outs_initializer +=
            paddle::string::Sprintf(OUT_INITIALIZER_TEMPLATE, out_name);
      }
      outs_initializer += ",";
    }

    // return_str += paddle::string::Sprintf(return_template, out_name);
    // return_str += ",";
    outs_num += 1;
  }
  call_api_str += "attrs);";
  if (outs_initializer.back() == ',') {
    outs_initializer.pop_back();
    // return_str.pop_back();
  }
  outs_initializer += "}";
  if (FindViewOpMap(op_type)) {
    std::string viwe_input_name = view_op_map[op_type].first;
    std::string viwe_output_name = view_op_map[op_type].second;
    view_strategy_str += paddle::string::Sprintf(
        HANDLE_VIEW_BETWEEN_INPUT_AND_OUTPUT, viwe_input_name, viwe_output_name,
        viwe_input_name, viwe_output_name);
  }
  if (outs_num == 0) {
    return_str = "Py_INCREF(Py_None);\n    return Py_None;";
  } else {
    return_str = "return ToPyObject(out);";
  }
  std::string function_args = "";
  if (input_args == "") {
    function_args = FUNCTION_ARGS_NO_INPUT;
  } else {
    function_args = paddle::string::Sprintf(FUNCTION_ARGS, input_args);
  }

  // generate op funtcion body
  auto op_function_str = paddle::string::Sprintf(
      OP_FUNCTION_TEMPLATE, func_name, ins_cast_str, op_type, input_args_num,
      call_api_str, return_str);

  return op_function_str;
}

static std::tuple<std::vector<std::string>, std::vector<std::string>>
GenerateOpFunctions() {
  auto& op_info_map = paddle::framework::OpInfoMap::Instance().map();

  std::vector<std::string> op_function_list, bind_function_list;
  auto& all_kernels = paddle::framework::OperatorWithKernel::AllOpKernels();

  for (auto& pair : op_info_map) {
    auto& op_info = pair.second;
    auto op_proto = op_info.proto_;
    if (op_proto == nullptr) {
      continue;
    }
    auto& op_type = op_proto->type();
    // Skip ooerator which is not inherit form OperatorWithKernel, like while,
    // since only OperatorWithKernel can run in dygraph mode.
    // if the pten lib contains op kernel, we still generate ops method
    if (!all_kernels.count(op_type) &&
        !pten::KernelFactory::Instance().HasCompatiblePtenKernel(op_type)) {
      continue;
    }
    if (!gen_list.count(op_type)) {
      continue;
    }
    std::string func_name = "eager_api_" + op_type;
    std::string op_function_str = GenerateOpFunctionsBody(op_proto, func_name);

    // generate pybind item
    auto bind_function_str = paddle::string::Sprintf(
        PYBIND_ITEM_TEMPLATE, op_type, func_name, op_type);

    op_function_list.emplace_back(std::move(op_function_str));
    bind_function_list.emplace_back(std::move(bind_function_str));
  }
  return std::make_tuple(op_function_list, bind_function_list);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "argc must be 2" << std::endl;
    return -1;
  }

#ifdef PADDLE_WITH_ASCEND_CL
  auto ascend_ptr = paddle::framework::AscendInstance::GetInstance();
  ascend_ptr->InitGEForUT();
#endif

  std::vector<std::string> headers{
      "\"pybind11/detail/common.h\"",
      "\"paddle/fluid/pybind/op_function_common.h\"",
      "\"paddle/fluid/pybind/exception.h\"", "<Python.h>"};

  std::ofstream out(argv[1], std::ios::out);

  out << "#pragma once\n\n";

  for (auto& header : headers) {
    out << "#include  " + header + "\n";
  }

  out << "\n\n";

  auto op_funcs = GenerateOpFunctions();

  out << "namespace paddle {\n"
      << "namespace pybind {\n\n";
  out << paddle::string::join_strings(std::get<0>(op_funcs), '\n');
  out << "\n\n";

  out << "static PyMethodDef ExtestMethods[] = {\n"
      << paddle::string::join_strings(std::get<1>(op_funcs), '\n')
      << "\n  {nullptr,nullptr,0,nullptr}"
      << "};\n\n";

  out << "inline void BindEagerOpFunctions(pybind11::module *module) {\n"
      << "  auto m = module->def_submodule(\"ops\");\n"
      << "  if (PyModule_AddFunctions(m.ptr(), ExtestMethods) < 0) {\n"
      << "    PADDLE_THROW(platform::errors::Fatal (\"Add functions to "
         "core.eager.ops failed!\"));\n"
      << "  }\n\n"
      << "  InitOpsAttrTypeMap();"
      << "}\n\n"
      << "} // namespace pybind\n"
      << "} // namespace paddle\n";

  out.close();

#ifdef PADDLE_WITH_ASCEND_CL
  ge::GEFinalize();
#endif

  return 0;
}
