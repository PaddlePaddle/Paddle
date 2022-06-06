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

KernelSignature LerpOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("lerp", {"X", "Y", "Weight"}, {}, {"Out"});
}

KernelSignature LerpGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("lerp_grad",
                         {"X", "Y", "Weight", "Out", "Out@GRAD"},
                         {},
                         {"X@GRAD", "Y@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(lerp, phi::LerpOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(lerp_grad, phi::LerpGradOpArgumentMapping);
