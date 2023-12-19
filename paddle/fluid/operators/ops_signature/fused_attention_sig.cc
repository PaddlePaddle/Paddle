/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

KernelSignature AttentionFuseOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("fused_attention",
                         {"X",
                          "LnScale",
                          "LnBias",
                          "QKVW",
                          "QKVBias",
                          "CacheKV",
                          "SrcMask",
                          "OutLinearW",
                          "OutLinearBias",
                          "Ln2Scale",
                          "Ln2Bias"},
                         {"num_heads",
                          "transpose_qkv_wb",
                          "pre_layer_norm",
                          "epsilon",
                          "attn_dropout_rate",
                          "is_test",
                          "attn_dropout_fix_seed",
                          "attn_dropout_seed",
                          "attn_dropout_implementation",
                          "dropout_rate",
                          "dropout_fix_seed",
                          "dropout_seed",
                          "dropout_implementation",
                          "ln_epsilon",
                          "add_residual",
                          "ring_id"},
                         {"LnMean",         "LnVariance",
                          "LnOut",          "QKVOut",
                          "QKVBiasOut",     "TransposeOut2",
                          "QKOut",          "QKTVOut",
                          "SoftmaxOut",     "AttnDropoutMaskOut",
                          "AttnDropoutOut", "SrcMaskOut",
                          "FMHAOut",        "OutLinearOut",
                          "DropoutMaskOut", "Ln2Mean",
                          "Ln2Variance",    "BiasDropoutResidualOut",
                          "CacheKVOut",     "Y"});
}

KernelSignature AttentionGradFuseOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("fused_attention_grad",
                         {"Y@GRAD",
                          "X",
                          "QKVW",
                          "QKVBias",
                          "QKVBiasOut",
                          "SrcMask",
                          "SrcMaskOut",
                          "OutLinearW",
                          "OutLinearBias",
                          "LnScale",
                          "LnBias",
                          "Ln2Scale",
                          "Ln2Bias",
                          "LnOut",
                          "LnMean",
                          "LnVariance",
                          "Ln2Mean",
                          "Ln2Variance",
                          "BiasDropoutResidualOut",
                          "QKVOut",
                          "TransposeOut2",
                          "QKOut",
                          "QKTVOut",
                          "SoftmaxOut",
                          "AttnDropoutMaskOut",
                          "AttnDropoutOut",
                          "FMHAOut",
                          "OutLinearOut",
                          "DropoutMaskOut"},
                         {"num_heads",
                          "transpose_qkv_wb",
                          "pre_layer_norm",
                          "epsilon",
                          "attn_dropout_rate",
                          "is_test",
                          "attn_dropout_fix_seed",
                          "attn_dropout_seed",
                          "attn_dropout_implementation",
                          "dropout_rate",
                          "dropout_fix_seed",
                          "dropout_seed",
                          "dropout_implementation",
                          "ln_epsilon",
                          "add_residual",
                          "ring_id"},
                         {
                             "QKVBias@GRAD",
                             "QKVBiasOut@GRAD",
                             "SrcMaskOut@GRAD",
                             "OutLinearBias@GRAD",
                             "LnScale@GRAD",
                             "LnBias@GRAD",
                             "Ln2Scale@GRAD",
                             "Ln2Bias@GRAD",
                             "X@GRAD",
                             "QKVW@GRAD",
                             "OutLinearW@GRAD",
                             "LnOut@GRAD",
                             "BiasDropoutResidualOut@GRAD",
                             "QKVOut@GRAD",
                             "QKTVOut@GRAD",
                             "TransposeOut2@GRAD",
                             "QKOut@GRAD",
                             "SoftmaxOut@GRAD",
                             "AttnDropoutOut@GRAD",
                             "FMHAOut@GRAD",
                             "OutLinearOut@GRAD",
                         });
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(fused_attention,
                           phi::AttentionFuseOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(fused_attention_grad,
                           phi::AttentionGradFuseOpArgumentMapping);
