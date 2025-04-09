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
#include <string>

#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/utils/small_vector.h"

namespace phi {

KernelSignature FusedAdamOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  paddle::small_vector<const char*> in_names = {"Params",
                                                "Grads",
                                                "LearningRate",
                                                "Moments1",
                                                "Moments2",
                                                "Moments2Max",
                                                "Beta1Pows",
                                                "Beta2Pows",
                                                "MasterParams",
                                                "SkipUpdate"};
  paddle::small_vector<const char*> out_names = {"ParamsOut",
                                                 "Moments1Out",
                                                 "Moments2Out",
                                                 "Moments2MaxOut",
                                                 "Beta1PowsOut",
                                                 "Beta2PowsOut",
                                                 "MasterParamsOut"};
  paddle::small_vector<const char*> attr_names = {"beta1",
                                                  "beta2",
                                                  "epsilon",
                                                  "chunk_size",
                                                  "weight_decay",
                                                  "use_adamw",
                                                  "multi_precision",
                                                  "use_global_beta_pow",
                                                  "amsgrad"};

  return KernelSignature("fused_adam",
                         std::move(in_names),
                         std::move(attr_names),
                         std::move(out_names));
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(fused_adam, phi::FusedAdamOpArgumentMapping);
