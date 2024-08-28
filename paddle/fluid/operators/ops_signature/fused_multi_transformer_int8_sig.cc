// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature FusedMultiTransformerInt8OpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  paddle::small_vector<const char*> inputs{
      "X",           "LnScale",           "LnBias",       "QKVW",
      "QKVBias",     "CacheKV",           "TimeStep",     "SrcMask",
      "OutLinearW",  "OutLinearBias",     "FFNLnScale",   "FFNLnBias",
      "FFN1Weight",  "FFN1Bias",          "FFN2Weight",   "FFN2Bias",
      "QKVOutScale", "OutLinearOutScale", "FFN1OutScale", "FFN2OutScale"};
  paddle::small_vector<const char*> attrs;
  attrs.emplace_back("pre_layer_norm");
  attrs.emplace_back("epsilon");
  attrs.emplace_back("dropout_rate");
  attrs.emplace_back("is_test");
  attrs.emplace_back("dropout_implementation");
  attrs.emplace_back("act_method");
  attrs.emplace_back("trans_qkvw");
  attrs.emplace_back("ring_id");
  attrs.emplace_back("num_head");
  attrs.emplace_back("dim_head");
  attrs.emplace_back("dim_ffn");
  attrs.emplace_back("qkv_in_scale");
  attrs.emplace_back("out_linear_in_scale");
  attrs.emplace_back("ffn1_in_scale");
  attrs.emplace_back("ffn2_in_scale");
  attrs.emplace_back("quant_round_type");
  attrs.emplace_back("quant_max_bound");
  attrs.emplace_back("quant_min_bound");
  paddle::small_vector<const char*> outputs{"CacheKVOut", "Out"};
  return KernelSignature("fused_multi_transformer_int8",
                         std::move(inputs),
                         std::move(attrs),
                         std::move(outputs));
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(fused_multi_transformer_int8,
                           phi::FusedMultiTransformerInt8OpArgumentMapping);
