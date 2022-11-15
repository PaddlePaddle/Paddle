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

KernelSignature BilinearTensorProductOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "bilinear_tensor_product", {"X", "Y", "Weight", "Bias"}, {}, {"Out"});
}

KernelSignature BilinearTensorProductGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("bilinear_tensor_product_grad",
                         {"X", "Y", "Weight", "Out@GRAD"},
                         {},
                         {"X@GRAD", "Y@GRAD", "Weight@GRAD", "Bias@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(bilinear_tensor_product,
                           phi::BilinearTensorProductOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(bilinear_tensor_product_grad,
                           phi::BilinearTensorProductGradOpArgumentMapping);
