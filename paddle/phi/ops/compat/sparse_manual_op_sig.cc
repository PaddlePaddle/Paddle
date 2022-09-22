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
      "sparse_coo_tensor", {"values", "indices"}, {"dense_shape"}, {"out"});
}

KernelSignature SparseValuesOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("x")) {
    return KernelSignature("values_coo", {"x"}, {}, {"out"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

KernelSignature SparseIndicesOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("x")) {
    return KernelSignature("indices_coo", {"x"}, {}, {"out"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

KernelSignature SparseToDenseOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("x")) {
    return KernelSignature("coo_to_dense", {"x"}, {}, {"out"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

KernelSignature SparseReluOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("x")) {
    return KernelSignature("relu_coo", {"x"}, {}, {"out"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

KernelSignature SparseConv3dOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("x")) {
    return KernelSignature(
        "conv3d_coo",
        {"x", "kernel"},
        {"paddings", "dilations", "strides", "groups", "subm", "key"},
        {"out", "rulebook", "counter"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

KernelSignature SparseAddOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsSparseCooTensorInput("x") && ctx.IsSparseCooTensorInput("y")) {
    return KernelSignature("add_coo_coo", {"x", "y"}, {}, {"out"});
  } else if (ctx.IsSparseCooTensorInput("x") && ctx.IsDenseTensorInput("y")) {
    return KernelSignature("add_coo_dense", {"x", "y"}, {}, {"out"});
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
