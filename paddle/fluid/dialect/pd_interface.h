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

#include "paddle/fluid/dialect/utils.h"
#include "paddle/ir/op_base.h"

using OpInfoTuple = std::tuple<std::vector<paddle::dialect::OpInputInfo>,
                               std::vector<paddle::dialect::OpAttributeInfo>,
                               std::vector<paddle::dialect::OpOutputInfo>>;

namespace paddle {
namespace dialect {
class GetOpInfoInterface : public ir::OpInterfaceBase<GetOpInfoInterface> {
 public:
  struct Concept {
    explicit Concept(OpInfoTuple (*get_op_info)(ir::Operation *))
        : get_op_info_(get_op_info) {}
    OpInfoTuple (*get_op_info_)(ir::Operation *);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static OpInfoTuple GetOpInfo(ir::Operation *op) {
      ConcreteOp concret_op = ConcreteOp(op);
      if (concret_op == nullptr) throw("concret_op is nullptr");
      return concret_op.GetOpInfo();
    }

    Model() : Concept(GetOpInfo) {
      static_assert(sizeof(Model) == sizeof(Concept),
                    "sizeof(Model) != sizeof(Concept)");
    }
  };

  GetOpInfoInterface(ir::Operation *op, Concept *impl)
      : ir::OpInterfaceBase<GetOpInfoInterface>(op), impl_(impl) {}

  OpInfoTuple GetOpInfo() { return impl_->get_op_info_(operation()); }

 private:
  Concept *impl_;
};

}  // namespace dialect
}  // namespace paddle
