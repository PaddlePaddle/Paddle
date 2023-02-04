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

#include "paddle/phi/core/operator_manager.h"

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

DECLARE_string(tensor_operator);

namespace paddle {

namespace experimental {

OperatorManager& OperatorManager::Instance() {
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

Tensor OperatorManager::multiply(const Tensor& x, const Tensor& y) {
  // VLOG(1) << "DEBUG OperatorManager begin x numel " << x.numel() << " y numel
  // " << y.numel();
  if (FLAGS_tensor_operator == "eager") {
    VLOG(1) << "DEBUG OperatorManager reach eager mode";
    PADDLE_ENFORCE_NE(this->eager_operator,
                      nullptr,
                      "eager mode, OperatorManager uses nullptr");

    Tensor result = this->eager_operator->multiply(x, y);
    VLOG(1) << "DEBUG OperatorManager finish eager mode";
    VLOG(1) << "tensor numel " << result.numel();
    return result;
  } else if (FLAGS_tensor_operator == "static") {
    VLOG(1) << "DEBUG OperatorManager reach static mode";
    PADDLE_ENFORCE_NE(this->static_operator,
                      nullptr,
                      "static mode, OperatorManager uses nullptr");

    Tensor result = this->static_operator->multiply(x, y);
    VLOG(1) << "DEBUG OperatorManager finish static mode";
    VLOG(1) << "tensor numel " << result.numel();
    return result;
  } else if (FLAGS_tensor_operator == "phi") {
    VLOG(1) << "DEBUG OperatorManager reach phi mode";
    PADDLE_ENFORCE_NE(
        this->phi_operator, nullptr, "phi mode, OperatorManager uses nullptr");

    Tensor result = this->phi_operator->multiply(x, y);
    VLOG(1) << "DEBUG OperatorManager finish phi mode";
    VLOG(1) << "tensor numel " << result.numel();
    return result;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "OperatorManager does not support the operator"));
  }
}

}  // namespace experimental
}  // namespace paddle
