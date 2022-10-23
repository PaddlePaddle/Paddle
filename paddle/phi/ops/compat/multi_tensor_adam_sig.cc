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

KernelSignature MultiTensorAdamOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  paddle::small_vector<const char*> in_names = {"Params",
                                                "Grads",
                                                "Moments1",
                                                "Moments2",
                                                "MasterParam",
                                                "Beta1Pow",
                                                "Beta2Pow",
                                                "LearningRate",
                                                "SkipUpdate"};
  paddle::small_vector<const char*> out_names = {"ParamsOut",
                                                 "Moments1Out",
                                                 "Moments2Out",
                                                 "MasterParamOut",
                                                 "Beta1PowOut",
                                                 "Beta2PowOut"};
  paddle::small_vector<const char*> attr_names = {"beta1",
                                                  "beta2",
                                                  "epsilon",
                                                  "compute_group_size",
                                                  "weight_decay",
                                                  "mode",
                                                  "multi_precision",
                                                  "use_global_beta_pow"};

  return KernelSignature("multi_tensor_adam",
                         std::move(in_names),
                         std::move(attr_names),
                         std::move(out_names));
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(multi_tensor_adam,
                           phi::MultiTensorAdamOpArgumentMapping);
