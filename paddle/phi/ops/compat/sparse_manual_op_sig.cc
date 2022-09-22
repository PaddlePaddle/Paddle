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

// TODO(zhangkaihuo): add csr op

KernelSignature SparseSparseCooTensorOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "sparse_coo_tensor", {"Values", "Indices"}, {"dense_shape"}, {"Out"});
}

KernelSignature SparseValuesOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("X")) {
    return KernelSignature("values_coo", {"X"}, {}, {"Out"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

KernelSignature SparseIndicesOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("X")) {
    return KernelSignature("indices_coo", {"X"}, {}, {"Out"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

KernelSignature SparseToDenseOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("X")) {
    return KernelSignature("coo_to_dense", {"X"}, {}, {"Out"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

KernelSignature SparseReluOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("X")) {
    return KernelSignature("relu_coo", {"X"}, {}, {"Out"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

KernelSignature SparseConv3dOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("X")) {
    return KernelSignature(
        "conv3d_coo",
        {"X", "Kernel"},
        {"paddings", "dilations", "strides", "groups", "subm", "key"},
        {"Out", "Rulebook", "Counter"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

KernelSignature SparseAddOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("X") && ctx.IsSparseCooTensorInput("Y")) {
    return KernelSignature("add_coo_coo", {"X", "Y"}, {}, {"Out"});
  } else if (ctx.IsSparseCooTensorInput("X") && ctx.IsDenseTensorInput("Y")) {
    return KernelSignature("add_coo_dense", {"X", "Y"}, {}, {"Out"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(sparse_sparse_coo_tensor, sparse_coo_tensor);
PD_REGISTER_ARG_MAPPING_FN(sparse_sparse_coo_tensor,
                           phi::SparseSparseCooTensorOpArgumentMapping);

PD_REGISTER_BASE_KERNEL_NAME(sparse_values, values_coo);
PD_REGISTER_ARG_MAPPING_FN(sparse_values, phi::SparseValuesOpArgumentMapping);

PD_REGISTER_BASE_KERNEL_NAME(sparse_indices, indices_coo);
PD_REGISTER_ARG_MAPPING_FN(sparse_indices, phi::SparseIndicesOpArgumentMapping);

PD_REGISTER_BASE_KERNEL_NAME(sparse_to_dense, coo_to_dense);
PD_REGISTER_ARG_MAPPING_FN(sparse_to_dense,
                           phi::SparseToDenseOpArgumentMapping);

PD_REGISTER_BASE_KERNEL_NAME(sparse_relu, relu_coo);
PD_REGISTER_ARG_MAPPING_FN(sparse_relu, phi::SparseReluOpArgumentMapping);

PD_REGISTER_BASE_KERNEL_NAME(sparse_conv3d, conv3d_coo);
PD_REGISTER_ARG_MAPPING_FN(sparse_conv3d, phi::SparseConv3dOpArgumentMapping);

PD_REGISTER_BASE_KERNEL_NAME(sparse_add, add_coo_coo);
PD_REGISTER_ARG_MAPPING_FN(sparse_add, phi::SparseAddOpArgumentMapping);
