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

KernelSignature MatmulOpArgumentMapping(const ArgumentMappingContext& ctx) {
  paddle::small_vector<const char*> inputs{"X", "Y"};
  paddle::small_vector<const char*> attrs;
  attrs.emplace_back("transpose_X");
  attrs.emplace_back("transpose_Y");
  attrs.emplace_back("alpha");

  paddle::small_vector<const char*> outputs{"Out"};
  return KernelSignature(
      "legacy_matmul", std::move(inputs), std::move(attrs), std::move(outputs));
}

KernelSignature MatmulGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  paddle::small_vector<const char*> inputs{"X", "Y", "Out@GRAD"};
  paddle::small_vector<const char*> attrs;
  attrs.emplace_back("transpose_X");
  attrs.emplace_back("transpose_Y");
  attrs.emplace_back("alpha");

  paddle::small_vector<const char*> outputs{"X@GRAD", "Y@GRAD"};
  return KernelSignature("legacy_matmul_grad",
                         std::move(inputs),
                         std::move(attrs),
                         std::move(outputs));
}

KernelSignature MatmulGradGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  paddle::small_vector<const char*> inputs{
      "X", "Y", "grad_out", "grad_x@GRAD", "grad_y@GRAD"};
  paddle::small_vector<const char*> attrs;
  attrs.emplace_back("transpose_X");
  attrs.emplace_back("transpose_Y");
  attrs.emplace_back("alpha");

  paddle::small_vector<const char*> outputs{
      "X@GRAD", "Y@GRAD", "grad_out@GRAD"};
  return KernelSignature("legacy_matmul_double_grad",
                         std::move(inputs),
                         std::move(attrs),
                         std::move(outputs));
}
}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(matmul, legacy_matmul);
PD_REGISTER_BASE_KERNEL_NAME(matmul_grad, legacy_matmul_grad);
PD_REGISTER_BASE_KERNEL_NAME(matmul_grad_grad, legacy_matmul_double_grad);

PD_REGISTER_ARG_MAPPING_FN(matmul, phi::MatmulOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(matmul_grad, phi::MatmulGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(matmul_grad_grad,
                           phi::MatmulGradGradOpArgumentMapping);
