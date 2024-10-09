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

KernelSignature NceOpArgumentMapping(const ArgumentMappingContext& ctx) {
  paddle::small_vector<const char*> inputs{"Input",
                                           "Label",
                                           "Weight",
                                           "Bias",
                                           "SampleWeight",
                                           "CustomDistProbs",
                                           "CustomDistAlias",
                                           "CustomDistAliasProbs"};
  paddle::small_vector<const char*> attrs;
  attrs.emplace_back("num_total_classes");
  attrs.emplace_back("custom_neg_classes");
  attrs.emplace_back("num_neg_samples");
  attrs.emplace_back("sampler");
  attrs.emplace_back("seed");
  attrs.emplace_back("is_sparse");
  attrs.emplace_back("remote_prefetch");
  attrs.emplace_back("is_test");
  paddle::small_vector<const char*> outputs{
      "Cost", "SampleLogits", "SampleLabels"};
  return KernelSignature(
      "nce", std::move(inputs), std::move(attrs), std::move(outputs));
}

KernelSignature NceGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  paddle::small_vector<const char*> inputs{"Input",
                                           "Label",
                                           "Bias",
                                           "Weight",
                                           "SampleLogits",
                                           "SampleLabels",
                                           "SampleWeight",
                                           "CustomDistProbs",
                                           "CustomDistAlias",
                                           "CustomDistAliasProbs",
                                           "Cost@GRAD"};
  paddle::small_vector<const char*> attrs;
  attrs.emplace_back("num_total_classes");
  attrs.emplace_back("custom_neg_classes");
  attrs.emplace_back("num_neg_samples");
  attrs.emplace_back("sampler");
  attrs.emplace_back("seed");
  attrs.emplace_back("is_sparse");
  attrs.emplace_back("remote_prefetch");
  attrs.emplace_back("is_test");
  paddle::small_vector<const char*> outputs{
      "Input@GRAD", "Bias@GRAD", "Weight@GRAD"};
  if (ctx.IsDenseTensorOutput("Weight@GRAD")) {
    return KernelSignature(
        "nce_grad", std::move(inputs), std::move(attrs), std::move(outputs));
  } else {
    return KernelSignature(
        "nce_sr_grad", std::move(inputs), std::move(attrs), std::move(outputs));
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(nce, phi::NceOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(nce_grad, phi::NceGradOpArgumentMapping);
