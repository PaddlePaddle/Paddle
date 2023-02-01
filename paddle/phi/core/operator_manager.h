//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "gflags/gflags.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/tensor_operator_base.h"

DECLARE_string(tensor_operator);

namespace paddle {

namespace experimental {
class Tensor;

class OperatorManager {
 public:
  static OperatorManager& Instance() {
    static OperatorManager g_op_manager;
    return g_op_manager;
  }

  // Tensor& add_(Tensor& x, const Tensor& y) {
  //   if (FLAGS_tensor_operator == "eager"){
  //     return eager_operator->add_(x, y);
  //   } else if (FLAGS_tensor_operator == "static"){
  //     return static_operator->add_(x, y);
  //   } else if (FLAGS_tensor_operator == "phi"){
  //     return phi_operator->add_(x, y);
  //   }
  // }

  Tensor multiply(const Tensor& x, const Tensor& y) {
    if (FLAGS_tensor_operator == "eager") {
      return eager_operator->multiply(x, y);
    } else if (FLAGS_tensor_operator == "static") {
      return static_operator->multiply(x, y);
    } else if (FLAGS_tensor_operator == "phi") {
      return phi_operator->multiply(x, y);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "OperatorManager does not support the operator "));
    }
  }

 public:
  TensorOperatorBase* eager_operator = nullptr;
  TensorOperatorBase* static_operator = nullptr;
  TensorOperatorBase* phi_operator = nullptr;

 private:
  OperatorManager() = default;
  DISABLE_COPY_AND_ASSIGN(OperatorManager);
};

}  // namespace experimental
}  // namespace paddle
