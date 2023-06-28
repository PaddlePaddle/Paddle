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

KernelSignature SyncBatchNormOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("sync_batch_norm",
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

KernelSignature SyncBatchNormGradOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("sync_batch_norm_grad",
                         {
                             "X",
                             "Scale",
                             "Bias",
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

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(sync_batch_norm,
                           phi::SyncBatchNormOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(sync_batch_norm_grad,
                           phi::SyncBatchNormGradOpArgumentMapping);
