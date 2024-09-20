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

#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/pir/include/core/op_base.h"

using OpInfoTuple = std::tuple<std::vector<paddle::dialect::OpInputInfo>,
                               std::vector<paddle::dialect::OpAttributeInfo>,
                               std::vector<paddle::dialect::OpOutputInfo>,
                               paddle::dialect::OpRunTimeInfo,
                               std::string>;

namespace paddle {
namespace dialect {
class OpYamlInfoInterface : public pir::OpInterfaceBase<OpYamlInfoInterface> {
 public:
  struct Concept {
    explicit Concept(OpInfoTuple (*get_op_info)(const std::string& op_name))
        : get_op_info_(get_op_info) {}
    OpInfoTuple (*get_op_info_)(const std::string& op_name);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static OpInfoTuple GetOpInfo(const std::string& op_name) {
      return ConcreteOp::GetOpInfo();
    }

    Model() : Concept(GetOpInfo) {}
  };

  /// Constructor
  OpYamlInfoInterface(const pir::Operation* op, Concept* impl)
      : pir::OpInterfaceBase<OpYamlInfoInterface>(op), impl_(impl) {}

  OpInfoTuple GetOpInfo() { return impl_->get_op_info_(operation_->name()); }

 private:
  Concept* impl_;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::OpYamlInfoInterface)
