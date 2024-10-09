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

#include "paddle/pir/include/core/op_base.h"

namespace paddle {
namespace dialect {
class VjpInterface : public pir::OpInterfaceBase<VjpInterface> {
 public:
  struct Concept {
    explicit Concept(std::vector<std::vector<pir::Value>> (*vjp)(
        pir::Operation* op,
        const std::vector<std::vector<pir::Value>>& inputs,
        const std::vector<std::vector<pir::Value>>& outputs,
        const std::vector<std::vector<pir::Value>>& out_grads,
        const std::vector<std::vector<bool>>& stop_gradients))
        : vjp_(vjp) {}
    std::vector<std::vector<pir::Value>> (*vjp_)(
        pir::Operation* op,
        const std::vector<std::vector<pir::Value>>& inputs,
        const std::vector<std::vector<pir::Value>>& outputs,
        const std::vector<std::vector<pir::Value>>& out_grads,
        const std::vector<std::vector<bool>>& stop_gradients);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static std::vector<std::vector<pir::Value>> Vjp(
        pir::Operation* op,
        const std::vector<std::vector<pir::Value>>& inputs,
        const std::vector<std::vector<pir::Value>>& outputs,
        const std::vector<std::vector<pir::Value>>& out_grads,
        const std::vector<std::vector<bool>>& stop_gradients) {
      return ConcreteOp::Vjp(op, inputs, outputs, out_grads, stop_gradients);
    }

    Model() : Concept(Vjp) {}
  };

  /// Constructor
  VjpInterface(const pir::Operation* op, Concept* impl)
      : pir::OpInterfaceBase<VjpInterface>(op), impl_(impl) {}

  std::vector<std::vector<pir::Value>> Vjp(
      pir::Operation* op,
      const std::vector<std::vector<pir::Value>>& inputs,
      const std::vector<std::vector<pir::Value>>& outputs,
      const std::vector<std::vector<pir::Value>>& out_grads,
      const std::vector<std::vector<bool>>& stop_gradients) {
    return impl_->vjp_(op, inputs, outputs, out_grads, stop_gradients);
  }

 private:
  Concept* impl_;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::VjpInterface)
