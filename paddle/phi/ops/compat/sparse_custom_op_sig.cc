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

KernelSignature SparseCooTensorOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "sparse_coo_tensor", {"Values", "Indices"}, {"dense_shape"}, {"Out"});
}

KernelSignature ValuesCooOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("values_coo", {"X"}, {}, {"Out"});
}

KernelSignature IndicesCooOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("indices_coo", {"X"}, {}, {"Out"});
}

KernelSignature CooToDenseOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("coo_to_dense", {"X"}, {}, {"Out"});
}

KernelSignature ReluCooOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("relu_coo", {"X"}, {}, {"Out"});
}

KernelSignature ShapeCooOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("shape_coo", {"X"}, {}, {"Out"});
}

KernelSignature Conv3dCooOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "conv3d_coo",
      {"X", "Kernel"},
      {"paddings", "dilations", "strides", "groups", "subm", "key"},
      {"Out", "Rulebook", "Counter"});
}

KernelSignature ValuesAddCooCooOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("values_add_coo_coo", {"X", "Y"}, {}, {"Out"});
}

KernelSignature ValuesAddCooDenseOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("values_add_coo_dense", {"X", "Y"}, {}, {"Out"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(sparse_coo_tensor,
                           phi::SparseCooTensorOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(values_coo, phi::ValuesCooOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(indices_coo, phi::IndicesCooOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(coo_to_dense, phi::CooToDenseOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(relu_coo, phi::ReluCooOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(shape_coo, phi::ShapeCooOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(conv3d_coo, phi::Conv3dCooOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(values_add_coo_coo,
                           phi::ValuesAddCooCooOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(values_add_coo_dense,
                           phi::ValuesAddCooDenseOpArgumentMapping);
