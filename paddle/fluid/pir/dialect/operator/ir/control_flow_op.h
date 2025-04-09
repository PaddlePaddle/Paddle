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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_symbolic_shape.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/interface/vjp.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/op_base.h"

namespace paddle {
namespace dialect {

class IR_API IfOp
    : public pir::Op<IfOp, VjpInterface, InferSymbolicShapeInterface> {
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

  static std::vector<std::vector<pir::Value>> Vjp(
      pir::Operation *op,
      const std::vector<std::vector<pir::Value>> &inputs_,
      const std::vector<std::vector<pir::Value>> &outputs,
      const std::vector<std::vector<pir::Value>> &out_grads,
      const std::vector<std::vector<bool>> &stop_gradients);

  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
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
class IR_API WhileOp
    : public pir::Op<WhileOp, VjpInterface, InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.while"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value cond,
                    const std::vector<pir::Value> &inputs,
                    bool construct_body = true);
  pir::Block &body();
  pir::Value cond();
  const pir::Block::ArgsType &block_args() { return body().args(); }
  void Print(pir::IrPrinter &printer);  // NOLINT
  void VerifySig();
  void VerifyRegion();
  static std::vector<std::vector<pir::Value>> Vjp(
      pir::Operation *op,
      const std::vector<std::vector<pir::Value>> &inputs_,
      const std::vector<std::vector<pir::Value>> &outputs,
      const std::vector<std::vector<pir::Value>> &out_grads,
      const std::vector<std::vector<bool>> &stop_gradients);
  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
};

struct TuplePushOpVjpInterfaceModel : public VjpInterface::Concept {
  static std::vector<std::vector<pir::Value>> Vjp(
      pir::Operation *op,
      const std::vector<std::vector<pir::Value>> &inputs,
      const std::vector<std::vector<pir::Value>> &outputs,
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
class HasElementsOp
    : public pir::Op<HasElementsOp, InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "cf.has_elements"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  TEST_API static void Build(pir::Builder &builder,             // NOLINT
                             pir::OperationArgument &argument,  // NOLINT
                             pir::Value container);
  void VerifySig();
  pir::Value input() { return operand_source(0); }
  pir::Value out() { return result(0); }
  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
};

///
/// \brief The AssertOp is an operation that asserts the given condition is
/// true. If the condition is false, prints the tensors in data. ``summarize``
/// specifies the number of the elements in the tensors to print.

/// It takes two inputs: cond and data, and takes one attribute: summarize. The
/// semantics of AssertOp[assert_op(cond, data, summarize)] are as below:
///   if(!cond){
///      print(summarize number of elements in data)
///   }
///
class AssertOp
    : public pir::Op<AssertOp, OpYamlInfoInterface, pir::SideEffectTrait> {
 public:
  using Op::Op;
  static const char ERROR_INFO_ATTR_NAME[];
  static const char *name() { return "pd_op.assert"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[1];

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value cond_,
                    pir::Value data_,
                    int64_t summarize);

  static OpInfoTuple GetOpInfo();
  void VerifySig();

  pir::Value cond() { return operand_source(0); }
  pir::Value data() { return operand_source(1); }
};

class SelectInputOp
    : public pir::Op<SelectInputOp, InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.select_input"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  void VerifySig();
  pir::Value mask() { return operand_source(0); }
  pir::Value out() { return result(0); }
  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
};

class SelectOutputOp : public pir::Op<SelectOutputOp> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.select_output"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  void VerifySig();
  pir::Value mask() { return operand_source(0); }
  pir::Value x() { return operand_source(1); }
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::IfOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::WhileOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::HasElementsOp);
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AssertOp);
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SelectInputOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SelectOutputOp)
