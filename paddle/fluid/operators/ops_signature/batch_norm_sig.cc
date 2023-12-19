// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

KernelSignature BatchNormOpArgumentMapping(const ArgumentMappingContext& ctx) {
  bool is_test = paddle::any_cast<bool>(ctx.Attr("is_test"));
  bool use_global_stats =
      ctx.HasAttr("use_global_stats")
          ? paddle::any_cast<bool>(ctx.Attr("use_global_stats"))
          : false;
  bool trainable_statistics =
      ctx.HasAttr("trainable_statistics")
          ? paddle::any_cast<bool>(ctx.Attr("trainable_statistics"))
          : false;
  bool fuse_with_relu = ctx.HasAttr("fuse_with_relu")
                            ? paddle::any_cast<bool>(ctx.Attr("fuse_with_relu"))
                            : false;
  // Dispenable `MomentumTensor` is useless now
  if (is_test && !use_global_stats && !trainable_statistics &&
      !fuse_with_relu) {
    return KernelSignature("batch_norm_infer",
                           {"X", "Mean", "Variance", "Scale", "Bias"},
                           {"momentum", "epsilon", "data_layout"},
                           {"Y", "MeanOut", "VarianceOut"});
  } else {
    return KernelSignature("batch_norm",
                           {"X", "Mean", "Variance", "Scale", "Bias"},
                           {"is_test",
                            "momentum",
                            "epsilon",
                            "data_layout",
                            "use_global_stats",
                            "trainable_statistics"},
                           {"Y",
                            "MeanOut",
                            "VarianceOut",
                            "SavedMean",
                            "SavedVariance",
                            "ReserveSpace"});
  }
}

KernelSignature BatchNormGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("batch_norm_grad",
                         {
                             "X",
                             "Scale",
                             "Bias",
                             "Mean",
                             "Variance",
                             "SavedMean",
                             "SavedVariance",
                             "ReserveSpace",
                             "Y@GRAD",
                         },
                         {"momentum",
                          "epsilon",
                          "data_layout",
                          "is_test",
                          "use_global_stats",
                          "trainable_statistics"},
                         {"X@GRAD", "Scale@GRAD", "Bias@GRAD"});
}

KernelSignature BatchNormGradGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("batch_norm_double_grad",
                         {"X",
                          "Scale",
                          "Mean",
                          "Variance",
                          "SavedMean",
                          "SavedVariance",
                          "DY",
                          "DDX",
                          "DDScale",
                          "DDBias"},
                         {"momentum",
                          "epsilon",
                          "data_layout",
                          "is_test",
                          "use_global_stats",
                          "trainable_statistics"},
                         {"DX", "DScale", "DDY"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(batch_norm, phi::BatchNormOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(batch_norm_grad,
                           phi::BatchNormGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(batch_norm_grad_grad,
                           phi::BatchNormGradGradOpArgumentMapping);
