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
#include <vector>

#include "paddle/fluid/pir/dialect/operator/interface/vjp.h"
#include "paddle/pir/core/op_base.h"

namespace paddle {
namespace dialect {

class IfOp : public pir::Op<IfOp, VjpInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.if"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value cond,
                    std::vector<pir::Type> &&output_types);

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value cond,
                    std::unique_ptr<pir::Block> &&true_block,
                    std::unique_ptr<pir::Block> &&false_block);

  pir::Value cond() { return operand_source(0); }
  pir::Block &true_block();
  pir::Block &false_block();
  pir::Region &true_region() { return (*this)->region(0); }
  pir::Region &false_region() { return (*this)->region(1); }
  void Print(pir::IrPrinter &printer);  // NOLINT
  void VerifySig();
  void VerifyRegion();

  static std::vector<std::vector<pir::OpResult>> Vjp(
      pir::Operation *op,
      const std::vector<std::vector<pir::Value>> &inputs_,
      const std::vector<std::vector<pir::OpResult>> &outputs,
      const std::vector<std::vector<pir::Value>> &out_grads,
      const std::vector<std::vector<bool>> &stop_gradients);
};

///
/// \brief The WhileOp is an operation that iterates over a loop body based on a
/// condition. It takes two inputs: cond_value and loop_vars. The output of the
/// WhileOp must have the same arity (length and structure) with loop_vars." The
/// semantics of WhileOp[outputs = while_op(cond, inputs)] are as below:
///   outputs = inputs
///   while(cond){
///      cond, outputs = body(outputs)
///   }
///
class WhileOp : public pir::Op<WhileOp> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.while"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value cond,
                    const std::vector<pir::Value> &inputs);
  pir::Block &body();
  pir::Value cond();
  void Print(pir::IrPrinter &printer);  // NOLINT
  void VerifySig() {}
  void VerifyRegion() {}
};

struct TuplePushOpVjpInterfaceModel : public VjpInterface::Concept {
  static std::vector<std::vector<pir::OpResult>> Vjp(
      pir::Operation *op,
      const std::vector<std::vector<pir::Value>> &inputs,
      const std::vector<std::vector<pir::OpResult>> &outputs,
      const std::vector<std::vector<pir::Value>> &out_grads,
      const std::vector<std::vector<bool>> &stop_gradients);

  TuplePushOpVjpInterfaceModel() : VjpInterface::Concept(Vjp) {}
};

///
/// \brief HasElementsOp is used in conjunction with WhileOp and StackType to
/// determine whether an element exists in the value corresponding to StackType
/// in the While usage scenario. Example:
/// (%stack_0, %inlet_0, %outlet_0) = "cf.create_stack" ()
/// ...
/// (%0) = "pd_op.has_elements" (%stack_0)
/// (...) = "pd_op.while"(%0) [...] {
///   ...
/// }
///
class HasElementsOp : public pir::Op<HasElementsOp> {
 public:
  using Op::Op;
  static const char *name() { return "cf.has_elements"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value stack);
  void VerifySig();
  pir::Value input() { return operand_source(0); }
  pir::Value out() { return result(0); }
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::IfOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::WhileOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::HasElementsOp);
