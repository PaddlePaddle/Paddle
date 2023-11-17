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

#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/dialect/control_flow/ir/cf_type.h"

namespace pir {

void YieldOp::Build(Builder &builder,
                    OperationArgument &argument,
                    const std::vector<Value> &inputs) {
  argument.AddInputs(inputs);
}

void CreateStackOp::Build(Builder &builder, OperationArgument &argument) {
  auto stack_type = StackType::get(builder.ir_context());
  auto inlet_type = InletType::get(builder.ir_context());
  auto outlet_type = OutletType::get(builder.ir_context());
  argument.AddOutputs({stack_type, inlet_type, outlet_type});
}

void CreateStackOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: CreateStackOp.";
  // Verify inputs:
  IR_ENFORCE(num_operands() == 0u, "The size of inputs must be equal to 0.");

  // No attributes should be verify.

  // Verify outputs:
  IR_ENFORCE(num_results() == 3u, "The size of outputs must be equal to 3.");

  IR_ENFORCE(result(0).type().isa<StackType>(),
             "The first outputs of cf.create_stack must be stack_type.");
  IR_ENFORCE(result(1).type().isa<InletType>(),
             "The first outputs of cf.create_stack must be inlet_type.");
  IR_ENFORCE(result(2).type().isa<OutletType>(),
             "The first outputs of cf.create_stack must be outlet_type.");

  VLOG(4) << "End Verifying for CreateStackOp.";
}

size_t CreateStackOp::stack_size() { return push_op().stack_size(); }

Value CreateStackOp::inlet_element(size_t index) {
  return push_op().inlet_element(index);
}

Value CreateStackOp::outlet_element(size_t index) {
  return pop_op().outlet_element(index);
}

PushBackOp CreateStackOp::push_op() {
  auto inlet_value = inlet();
  IR_ENFORCE(inlet_value.HasOneUse(), "The inlet value must has one use.");
  return inlet_value.first_use().owner()->dyn_cast<PushBackOp>();
}

PopBackOp CreateStackOp::pop_op() {
  auto outlet_value = outlet();
  IR_ENFORCE(outlet_value.HasOneUse(), "The outlet value must has one use.");
  return outlet_value.first_use().owner()->dyn_cast<PopBackOp>();
}

void CreateStackOp::Print(IrPrinter &printer) {  // NOLINT
  static std::unordered_map<IrPrinter *,
                            std::unordered_map<Operation *, size_t>>
      kConunters;
  auto &counter = kConunters[&printer];
  auto iter = counter.insert({*this, counter.size()});
  auto index = iter.first->second;
  if (iter.second) {
    printer.AddValueAlias(stack(), "%stack_" + std::to_string(index));
    printer.AddValueAlias(inlet(), "%inlet_" + std::to_string(index));
    printer.AddValueAlias(outlet(), "%outlet_" + std::to_string(index));
  }
  printer.PrintGeneralOperation(*this);
}

void PushBackOp::Build(Builder &builder,             // NOLINT
                       OperationArgument &argument,  // NOLINT
                       Value inlet,
                       const std::vector<Value> &elements) {
  argument.AddInput(inlet);
  argument.AddInputs(elements);
}

void PushBackOp::Build(Builder &builder,             // NOLINT
                       OperationArgument &argument,  // NOLINT
                       Value inlet,
                       std::initializer_list<Value> element_list) {
  argument.AddInput(inlet);
  argument.AddInputs(element_list);
}

void PushBackOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs ,attributes for: PushBackOp.";
  // Verify inputs:
  IR_ENFORCE(num_operands() >= 2u, "The size of inputs must no less than 2.");
  IR_ENFORCE(operand_source(0).type().isa<InletType>(),
             "The first input of cf.push_back must be inlet_type.");
  IR_ENFORCE(operand_source(0).HasOneUse(),
             "The inlet value of cf.push_back can only be used once.");

  // No attributes should be verify.

  // Verify outputs:
  IR_ENFORCE(num_results() == 0u, "The size of outputs must be equal to 0.");
  VLOG(4) << "End Verifying for PushBackOp.";
}

size_t PushBackOp::stack_size() {
  auto operands_size = num_operands();
  IR_ENFORCE(operands_size >= 2u,
             "The operands of push op must no less than 2.");
  return operands_size - 1u;
}

PopBackOp PushBackOp::pop_op() { return create_op().pop_op(); }

void PopBackOp::Build(Builder &builder,             // NOLINT
                      OperationArgument &argument,  // NOLINT
                      Value outlet) {
  argument.AddInput(outlet);

  auto push_back_op = outlet.defining_op<CreateStackOp>().push_op();

  auto elements_size = push_back_op.stack_size();

  for (size_t index = 0; index < elements_size; ++index) {
    argument.AddOutput(push_back_op.inlet_element(index).type());
  }
}

void PopBackOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs ,attributes  and stack validity for: "
             "PopBackOp.";
  // Verify inputs:
  IR_ENFORCE(num_operands() == 1u, "The size of inputs must equal to 1.");
  IR_ENFORCE(operand_source(0).type().isa<OutletType>(),
             "The first input of cf.pop_back must be outlet_type.");
  IR_ENFORCE(operand_source(0).HasOneUse(),
             "The outlet value of cf.pop_back can only be used once.");

  // No attributes should be verify.

  // Verify outputs:
  IR_ENFORCE(num_results() >= 1u,
             "The size of outputs must no less than to 1.");
  // Verify stack validity:
  auto pop_back_op = create_op().pop_op();
  IR_ENFORCE(*this == pop_back_op,
             "The pop_op of stack_op must be this pop_op self.");

  auto inlet_size = push_op().stack_size();
  IR_ENFORCE(inlet_size == stack_size(),
             "The pop elements size must equal to push elements size.");
  for (size_t index = 0; index < inlet_size; ++index) {
    IR_ENFORCE(outlet_element(index).type() == inlet_element(index).type(),
               "The %d element's push type (%s) isn't equal to pop type (%s)",
               index,
               outlet_element(index).type(),
               inlet_element(index).type());
  }
  VLOG(4) << "End Verifying for PopBackOp.";
}

}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::YieldOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::CreateStackOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::PushBackOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::PopBackOp)
