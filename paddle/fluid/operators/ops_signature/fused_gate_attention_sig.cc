/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature FusedGateAttentionOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  paddle::small_vector<const char*> inputs{"Query",
                                           "Key",
                                           "QueryWeight",
                                           "KeyWeight",
                                           "ValueWeight",
                                           "QKVWeight",
                                           "NonbatchedBias",
                                           "SrcMask",
                                           "GateWeight",
                                           "GateBias",
                                           "OutLinearWeight",
                                           "OutLinearBias"};
  paddle::small_vector<const char*> attrs;
  attrs.emplace_back("has_gating");
  attrs.emplace_back("merge_qkv");
  attrs.emplace_back("use_flash_attn");
  paddle::small_vector<const char*> outputs{"QueryTransposeOut",
                                            "KeyTransposeOut",
                                            "ValueTransposeOut",
                                            "QKVTransposeOut",
                                            "SoftmaxOut",
                                            "SoftmaxLse",
                                            "FMHAOut",
                                            "GateOut",
                                            "Out"};
  return KernelSignature("fused_gate_attention",
                         std::move(inputs),
                         std::move(attrs),
                         std::move(outputs));
}

KernelSignature FusedGateAttentionGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  paddle::small_vector<const char*> inputs{"Query",
                                           "Key",
                                           "QueryWeight",
                                           "KeyWeight",
                                           "ValueWeight",
                                           "QKVWeight",
                                           "NonbatchedBias",
                                           "SrcMask",
                                           "GateWeight",
                                           "GateBias",
                                           "OutLinearWeight",
                                           "OutLinearBias",
                                           "QueryTransposeOut",
                                           "KeyTransposeOut",
                                           "ValueTransposeOut",
                                           "QKVTransposeOut",
                                           "SoftmaxOut",
                                           "SoftmaxLse",
                                           "FMHAOut",
                                           "GateOut",
                                           "Out@GRAD"};
  paddle::small_vector<const char*> attrs;
  attrs.emplace_back("has_gating");
  attrs.emplace_back("merge_qkv");
  attrs.emplace_back("use_flash_attn");
  paddle::small_vector<const char*> outputs{"Query@GRAD",
                                            "Key@GRAD",
                                            "QueryWeight@GRAD",
                                            "KeyWeight@GRAD",
                                            "ValueWeight@GRAD",
                                            "QKVWeight@GRAD",
                                            "NonbatchedBias@GRAD",
                                            "GateWeight@GRAD",
                                            "GateBias@GRAD",
                                            "OutLinearWeight@GRAD",
                                            "OutLinearBias@GRAD"};
  return KernelSignature("fused_gate_attention_grad",
                         std::move(inputs),
                         std::move(attrs),
                         std::move(outputs));
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(fused_gate_attention,
                           phi::FusedGateAttentionOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(fused_gate_attention_grad,
                           phi::FusedGateAttentionGradOpArgumentMapping);
