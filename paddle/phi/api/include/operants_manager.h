// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/api/include/operants_base.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/macros.h"

namespace paddle {

using Tensor = paddle::experimental::Tensor;
using TensorOperantsBase = paddle::operants::TensorOperantsBase;

/**
 * [ Why need OperantsManager? ]
 *
 * Ideally, overloading tensor operators should call Tensor API directly.
 * However, we faced two problems:
 *
 * 1. Support multiple modes: Tensor operator overloading needs to support
 * [static mode / autograd mode / custom operator mode] at the same time.
 *
 * 2. Decouple phi and fluid: Tensor belongs to the phi library, but it relies
 * upon functions in fluid when overloading Tensor operators.
 *
 * We design OperantsManager to solve these two problems:
 *
 * 1. use `FLAGS_tensor_operants_mode` to handle overloading mode, set this flag
 * at the entry point of each mode:
 *
 * - FLAGS_tensor_operants_mode = "static": at the construction function of
 * `CompositeGradOpMakerBase`.
 * - FLAGS_tensor_operants_mode = "eager": at the beginning of dygraph_function.
 * - FLAGS_tensor_operants_mode = "phi": at the beginning of the
 * `eager_api_run_custom_op` function in eager mode and at the beginning of
 * calling kernels in static mode.
 *
 * In order to guarantee the performance, OperantsManager holds three pointers
 * to identify each mode respectively.
 *
 * 2. Decouple phi with the help of the polymorphism mechanism,
 * TensorOperantsBase derives three child classes: PhiTensorOperants,
 * EagerTensorOperants, and StaticTensorOperants. We set eager and static tensor
 * operants at the fluid library and set phi operants at the phi library.
 *
 */
class OperantsManager {
 public:
  static OperantsManager& Instance();

  Tensor multiply(const Tensor& x, const Tensor& y);

 public:
  std::unique_ptr<TensorOperantsBase> eager_operants{nullptr};
  std::unique_ptr<TensorOperantsBase> static_operants{nullptr};
  std::unique_ptr<TensorOperantsBase> phi_operants{nullptr};

 private:
  OperantsManager() = default;
  DISABLE_COPY_AND_ASSIGN(OperantsManager);
};

}  // namespace paddle
