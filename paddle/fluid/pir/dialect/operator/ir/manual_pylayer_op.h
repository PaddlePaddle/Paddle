// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/op_base.h"

namespace paddle {
namespace dialect {

class PyLayerOp : public pir::Op<PyLayerOp> {
 public:
  using Op::Op;

  static const char *name() { return "pd_op.pylayer"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Value> &inputs,
                    std::vector<pir::Type> &&output_types);

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Value> &inputs,
                    std::unique_ptr<pir::Block> &&fwd_block);

  std::vector<pir::Value> inputs() {
    std::vector<pir::Value> input_values;
    for (size_t index = 0; index < num_operands(); ++index) {
      input_values.push_back(operand_source(index));
    }
    return input_values;
  }
  pir::Value input(size_t index) {
    PADDLE_ENFORCE_LT(
        index,
        num_operands(),
        phi::errors::InvalidArgument("The index of input must be less than "
                                     "num_operands of pylayer op."));
    return operand_source(index);
  }
  pir::Block &forward_block();
  pir::Region &forward_region() { return (*this)->region(0); }

  void Print(pir::IrPrinter &printer);  // NOLINT
  void VerifySig();
  void VerifyRegion();

  void UpdateOutput();
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::PyLayerOp);
