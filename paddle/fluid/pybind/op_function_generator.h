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

#pragma once

#include <map>
#include <set>
#include <string>

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
    {"bincount", {"X", "Weights"}},
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
    {"repeat_interleave", {"X", "RepeatsTensor"}},
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
    {"merged_momentum",
     {"Param", "Grad", "Velocity", "LearningRate", "MasterParam"}},
    {"sparse_momentum", {"Param", "Grad", "Velocity", "Index", "LearningRate"}},
    {"rnn", {"Input", "PreState", "WeightList", "SequenceLength"}},
    {"run_program", {"X", "Params"}},
    {"fused_feedforward",
     {"Dropout1Seed", "Dropout2Seed", "Linear1Bias", "Linear2Bias", "Ln1Scale",
      "Ln1Bias", "Ln2Scale", "Ln2Bias"}},
    {"faster_tokenizer", {"Text", "Vocab", "TextPair"}},
    {"matrix_rank", {"X", "TolTensor"}},
    {"adam",
     {"Param", "Grad", "LearningRate", "Moment1", "Moment2", "Beta1Pow",
      "Beta2Pow", "MasterParam"}},
    {"merged_adam",
     {"Param", "Grad", "LearningRate", "Moment1", "Moment2", "Beta1Pow",
      "Beta2Pow", "MasterParam"}},
    {"adamw",
     {"Param", "Grad", "LearningRate", "Moment1", "Moment2", "Beta1Pow",
      "Beta2Pow", "MasterParam"}},
    {"lamb",
     {"Param", "Grad", "LearningRate", "Moment1", "Moment2", "Beta1Pow",
      "Beta2Pow", "MasterParam"}},
    {"sparse_attention",
     {"Q", "K", "V", "Offset", "Columns", "KeyPaddingMask", "AttnMask"}},
    {"sgd", {"Param", "LearningRate", "Grad", "MasterParam"}},
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
    {"merged_momentum", {"ParamOut", "VelocityOut", "MasterParamOut"}},
    {"sparse_momentum", {"ParamOut", "VelocityOut"}},
    {"rnn", {"DropoutState", "Reserve", "Out", "State"}},
    {"run_program", {"DOut"}},
    {"adam",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut",
      "MasterParamOut"}},
    {"merged_adam",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut",
      "MasterParamOut"}},
    {"adamw",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut",
      "MasterParamOut"}},
    {"sgd", {"ParamOut", "MasterParamOut"}},
    {"lamb",
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
    {"sgd", {"ParamOut", "MasterParamOut"}},
    {"rmsprop", {"ParamOut", "MomentOut", "MeanSquareOut", "MeanGradOut"}},
    {"ftrl", {"ParamOut", "SquaredAccumOut", "LinearAccumOut"}},
    {"adadelta", {"ParamOut", "AvgSquaredGradOut", "AvgSquaredUpdateOut"}},
    {"adagrad", {"ParamOut", "MomentOut"}},
    {"adamax", {"ParamOut", "MomentOut", "InfNormOut"}},
    {"dpsgd", {"ParamOut"}},
    {"decayed_adagrad", {"ParamOut", "MomentOut"}},
    {"lars_momentum", {"ParamOut", "VelocityOut"}},
    {"coalesce_tensor", {"Output", "FusedOutput"}},
    {"adam",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut",
      "MasterParamOut"}},
    {"merged_adam",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut",
      "MasterParamOut"}},
    {"adamw",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut",
      "MasterParamOut"}},
    {"lamb",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut",
      "MasterParamOut"}},
    {"average_accumulates",
     {"out_sum_1", "out_sum_2", "out_sum_3", "out_num_accumulates",
      "out_old_num_accumulates", "out_num_updates"}},
    {"momentum", {"ParamOut", "VelocityOut", "MasterParamOut"}},
    {"merged_momentum", {"ParamOut", "VelocityOut", "MasterParamOut"}},
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
