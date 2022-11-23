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

KernelSignature IndexAddOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "index_add", {"X", "Index", "AddValue"}, {"axis"}, {"Out"});
}

KernelSignature IndexAddGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("index_add_grad",
                         {"Index", "AddValue", "Out@GRAD"},
                         {"axis"},
                         {"X@GRAD", "AddValue@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(index_add, phi::IndexAddOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(index_add_grad, phi::IndexAddGradOpArgumentMapping);
