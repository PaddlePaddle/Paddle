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

#include "paddle/phi/api/include/operants_manager.h"

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

DECLARE_string(tensor_operants_mode);

namespace paddle {

OperantsManager& OperantsManager::Instance() {
  static OperantsManager g_op_manager;
  return g_op_manager;
}

Tensor OperantsManager::multiply(const Tensor& x, const Tensor& y) {
  if (FLAGS_tensor_operants_mode == "eager") {
    PADDLE_ENFORCE_NE(
        this->eager_operants.get(),
        nullptr,
        phi::errors::Unavailable("The eager_operants pointer of "
                                 "OperantsManager is not initialized"));
    VLOG(4) << "OperantsManager reaches eager mode";
    return this->eager_operants->multiply(x, y);
  } else if (FLAGS_tensor_operants_mode == "static") {
    PADDLE_ENFORCE_NE(
        this->static_operants.get(),
        nullptr,
        phi::errors::Unavailable("The static_operants pointer of "
                                 "OperantsManager is not initialized"));
    VLOG(4) << "OperantsManager reaches static mode";
    return this->static_operants->multiply(x, y);
  } else if (FLAGS_tensor_operants_mode == "phi") {
    PADDLE_ENFORCE_NE(
        this->phi_operants.get(),
        nullptr,
        phi::errors::Unavailable(
            "The phi_operants pointer of OperantsManager is not initialized"));
    VLOG(4) << "OperantsManager reaches phi mode";
    return this->phi_operants->multiply(x, y);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "FLAGS_tensor_operants_mode is not nitialized, please set "
        "FLAGS_tensor_operants_mode first, which currently supports eager, "
        "phi, and static mode"));
  }
}

}  // namespace paddle
