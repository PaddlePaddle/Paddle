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
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/op_base.h"

namespace paddle {
namespace dialect {

class PyLayerOp : public pir::Op<PyLayerOp> {
 public:
  using Op::Op;

  static const char *name() { return "pd_op.pylayer"; }
  static constexpr char kBackwardFunctionIdAttrName[] = "backward_function_id";
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Value> &inputs,
                    std::vector<pir::Type> &&output_types,
                    int backward_function_id = -1);

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Value> &inputs,
                    std::unique_ptr<pir::Block> &&fwd_block,
                    int backward_function_id = -1);

  std::vector<pir::Value> inputs() const {
    std::vector<pir::Value> input_values;
    for (size_t index = 0; index < num_operands(); ++index) {
      input_values.push_back(operand_source(index));
    }
    return input_values;
  }
  pir::Value input(size_t index) const {
    PADDLE_ENFORCE_LT(
        index,
        num_operands(),
        common::errors::InvalidArgument("The index of input must be less than "
                                        "num_operands of pylayer op."));
    return operand_source(index);
  }

  std::vector<pir::Value> outputs() const {
    std::vector<pir::Value> output_values;
    for (size_t index = 0; index < num_results(); ++index) {
      output_values.push_back(result(index));
    }
    return output_values;
  }

  pir::Block &forward_block();
  pir::Region &forward_region() { return (*this)->region(0); }

  // Returns the backward function id which may have been registered in
  // PythonCallableRegistrar. Returns -1 only if this PyLayer Op does not have a
  // backward function.
  int backward_function_id() const {
    return this->attributes()
        .at("backward_function_id")
        .dyn_cast<pir::Int32Attribute>()
        .data();
  }

  void Print(pir::IrPrinter &printer);  // NOLINT
  void VerifySig();
  void VerifyRegion();

  void UpdateOutput();
  PyLayerOp UpdateInput();
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::PyLayerOp);
