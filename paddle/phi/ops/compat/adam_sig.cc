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
  paddle::SmallVector<std::string> in_names = {"Param",
                                               "Grad",
                                               "LearningRate",
                                               "Moment1",
                                               "Moment2",
                                               "Beta1Pow",
                                               "Beta2Pow",
                                               "MasterParam",
                                               "SkipUpdate"};
  paddle::SmallVector<std::string> out_names = {"ParamOut",
                                                "Moment1Out",
                                                "Moment2Out",
                                                "Beta1PowOut",
                                                "Beta2PowOut",
                                                "MasterParamOut"};
  paddle::SmallVector<std::string> attr_names;

  attr_names.emplace_back(ctx.HasInput("Beta1Tensor") ? "Beta1Tensor"
                                                      : "beta1");
  attr_names.emplace_back(ctx.HasInput("Beta2Tensor") ? "Beta2Tensor"
                                                      : "beta2");
  attr_names.emplace_back(ctx.HasInput("EpsilonTensor") ? "EpsilonTensor"
                                                        : "epsilon");
  attr_names.emplace_back("lazy_mode");
  attr_names.emplace_back("min_row_size_to_use_multithread");
  attr_names.emplace_back("multi_precision");
  attr_names.emplace_back("use_global_beta_pow");

  if (ctx.IsSelectedRowsInput("Grad")) {
    return KernelSignature("adam_dense_param_sparse_grad",
                           std::move(in_names),
                           std::move(attr_names),
                           std::move(out_names));
  } else if (ctx.IsDenseTensorInput("Grad")) {
    return KernelSignature("adam",
                           std::move(in_names),
                           std::move(attr_names),
                           std::move(out_names));
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(adam, phi::AdamOpArgumentMapping);
