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

KernelSignature DropoutOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "dropout",
      {"X", "Seed"},
      {"dropout_prob", "is_test", "dropout_implementation", "seed", "fix_seed"},
      {"Out", "Mask"});
}

KernelSignature DropoutGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("dropout_grad",
                         {"Mask", "Out@GRAD"},
                         {"dropout_prob", "is_test", "dropout_implementation"},
                         {"X@GRAD"});
}

KernelSignature DropoutNdOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("dropout_nd",
                         {"X", "Seed"},
                         {"dropout_prob",
                          "is_test",
                          "dropout_implementation",
                          "seed",
                          "fix_seed",
                          "axis"},
                         {"Out", "Mask"});
}

KernelSignature DropoutNdGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "dropout_nd_grad",
      {"Mask", "Out@GRAD"},
      {"dropout_prob", "is_test", "dropout_implementation", "axis"},
      {"X@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(dropout, phi::DropoutOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(dropout_grad, phi::DropoutGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(dropout_nd, phi::DropoutNdOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(dropout_nd_grad,
                           phi::DropoutNdGradOpArgumentMapping);
