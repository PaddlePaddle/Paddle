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
class DecompInterface : public pir::OpInterfaceBase<DecompInterface> {
 public:
  struct Concept {
    explicit Concept(
        std::vector<std::vector<pir::Value>> (*decomp)(pir::Operation* op))
        : decomp_(decomp) {}
    std::vector<std::vector<pir::Value>> (*decomp_)(pir::Operation* op);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static std::vector<std::vector<pir::Value>> Decomp(pir::Operation* op) {
      return ConcreteOp::Decomp(op);
    }
    Model() : Concept(Decomp) {}
  };

  /// Constructor
  DecompInterface(const pir::Operation* op, Concept* impl)
      : pir::OpInterfaceBase<DecompInterface>(op), impl_(impl) {}

  std::vector<std::vector<pir::Value>> Decomp(pir::Operation* op) {
    return impl_->decomp_(op);
  }

 private:
  Concept* impl_;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DecompInterface)
