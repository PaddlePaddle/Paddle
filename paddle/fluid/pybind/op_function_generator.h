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
    {"fused_feedforward",
     {"Dropout1Seed", "Dropout2Seed", "Linear1Bias", "Linear2Bias", "Ln1Scale",
      "Ln1Bias", "Ln2Scale", "Ln2Bias"}},
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
