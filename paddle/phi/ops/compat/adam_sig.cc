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

KernelSignature AdamOpArgumentMapping(const ArgumentMappingContext& ctx) {
  paddle::small_vector<const char*> inputs{"Param",
                                           "Grad",
                                           "LearningRate",
                                           "Moment1",
                                           "Moment2",
                                           "Beta1Pow",
                                           "Beta2Pow",
                                           "MasterParam",
                                           "SkipUpdate"};
  paddle::small_vector<const char*> attrs;
  attrs.emplace_back(ctx.HasInput("Beta1Tensor") ? "Beta1Tensor" : "beta1");
  attrs.emplace_back(ctx.HasInput("Beta2Tensor") ? "Beta2Tensor" : "beta2");
  attrs.emplace_back(ctx.HasInput("EpsilonTensor") ? "EpsilonTensor"
                                                   : "epsilon");
  attrs.emplace_back("lazy_mode");
  attrs.emplace_back("min_row_size_to_use_multithread");
  attrs.emplace_back("multi_precision");
  attrs.emplace_back("use_global_beta_pow");
  paddle::small_vector<const char*> outputs{"ParamOut",
                                            "Moment1Out",
                                            "Moment2Out",
                                            "Beta1PowOut",
                                            "Beta2PowOut",
                                            "MasterParamOut"};
  if (ctx.IsDenseTensorInput("Param") && ctx.IsDenseTensorInput("Grad") &&
      ctx.IsDenseTensorInput("LearningRate") &&
      ctx.IsDenseTensorInput("Moment1") && ctx.IsDenseTensorInput("Moment2") &&
      ctx.IsDenseTensorInput("Beta1Pow") &&
      ctx.IsDenseTensorInput("Beta2Pow") &&
      ((ctx.HasInput("MasterParam") && ctx.IsDenseTensorInput("MasterParam")) ||
       (!ctx.HasInput("MasterParam"))) &&
      ((ctx.HasInput("SkipUpdate") && ctx.IsDenseTensorInput("SkipUpdate")) ||
       (!ctx.HasInput("SkipUpdate")))) {
    return KernelSignature(
        "adam", std::move(inputs), std::move(attrs), std::move(outputs));
  } else if (ctx.IsDenseTensorInput("Param") &&
             ctx.IsSelectedRowsInput("Grad") &&
             ctx.IsDenseTensorInput("LearningRate") &&
             ctx.IsDenseTensorInput("Moment1") &&
             ctx.IsDenseTensorInput("Moment2") &&
             ctx.IsDenseTensorInput("Beta1Pow") &&
             ctx.IsDenseTensorInput("Beta2Pow") &&
             ((ctx.HasInput("MasterParam") &&
               ctx.IsDenseTensorInput("MasterParam")) ||
              (!ctx.HasInput("MasterParam"))) &&
             ((ctx.HasInput("SkipUpdate") &&
               ctx.IsDenseTensorInput("SkipUpdate")) ||
              (!ctx.HasInput("SkipUpdate")))) {
    return KernelSignature("adam_dense_param_sparse_grad",
                           std::move(inputs),
                           std::move(attrs),
                           std::move(outputs));
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(adam, phi::AdamOpArgumentMapping);
