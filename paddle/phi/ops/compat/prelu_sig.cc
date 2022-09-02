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

KernelSignature PReluOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "prelu", {"X", "Alpha"}, {"data_format", "mode"}, {"Out"});
}

KernelSignature PReluGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("prelu_grad",
                         {"X", "Alpha", "Out@GRAD"},
                         {"data_format", "mode"},
                         {"X@GRAD", "Alpha@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(prelu, phi::PReluOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(prelu_grad, phi::PReluGradOpArgumentMapping);
