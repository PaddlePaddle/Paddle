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

KernelSignature GatherNdGradArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "gather_nd_grad", {"X", "Index", "Out@GRAD"}, {}, {"X@GRAD"});
}

KernelSignature ScatterGradArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("scatter_grad",
                         {"Ids", "Updates", "Out@GRAD"},
                         {"overwrite"},
                         {"X@GRAD", "Updates@GRAD"});
}

KernelSignature ScatterNdAddGradArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("scatter_nd_add_grad",
                         {"Index", "Updates", "Out@GRAD"},
                         {},
                         {"X@GRAD", "Updates@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(gather_nd_grad, phi::GatherNdGradArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(scatter_grad, phi::ScatterGradArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(scatter_nd_add_grad,
                           phi::ScatterNdAddGradArgumentMapping);
