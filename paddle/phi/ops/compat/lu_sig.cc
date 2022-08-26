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

KernelSignature LUOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("lu", {"X"}, {"pivots"}, {"Out", "Pivots", "Infos"});
}

KernelSignature LUGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "lu_grad", {"X", "Out", "Pivots", "Out@GRAD"}, {"pivots"}, {"X@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(lu, phi::LUOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(lu_grad, phi::LUGradOpArgumentMapping);
