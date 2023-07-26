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

#include "paddle/ir/core/op_base.h"

namespace paddle {
namespace dialect {
class VjpInterface : public ir::OpInterfaceBase<VjpInterface> {
 public:
  struct Concept {
    explicit Concept(std::vector<std::vector<ir::Value>> (*vjp)(
        std::vector<std::vector<ir::Value>> out_grads,
        const std::vector<std::vector<int>>& stop_gradients))
        : vjp_(vjp) {}
    std::vector<std::vector<ir::Value>> (*vjp_)(
        std::vector<std::vector<ir::Value>> out_grads,
        const std::vector<std::vector<int>>& stop_gradients);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static std::vector<std::vector<ir::Value>> Vjp(
        std::vector<std::vector<ir::Value>> out_grads,
        const std::vector<std::vector<int>>& stop_gradients) {
      return ConcreteOp::Vjp(out_grads, stop_gradients);
    }

    Model() : Concept(Vjp) {}
  };

  VjpInterface(ir::Operation* op, Concept* impl)
      : ir::OpInterfaceBase<VjpInterface>(op), impl_(impl) {}

  std::vector<std::vector<ir::Value>> Vjp(
      std::vector<std::vector<ir::Value>> out_grads,
      const std::vector<std::vector<int>>& stop_gradients) {
    return impl_->vjp_(out_grads, stop_gradients);
  }

 private:
  Concept* impl_;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::VjpInterface)
