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

#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/phi/core/enforce.h"

namespace ir {

const char *ModuleOp::attributes_name[attributes_num] = {"program"};

Program *ModuleOp::program() {
  const AttributeMap &attr = this->attributes();
  auto iter = attr.find("program");
  if (iter == attr.end() || !iter->second) return nullptr;
  return static_cast<Program *>(
      iter->second.dyn_cast<PointerAttribute>().data());
}

Block *ModuleOp::block() {
  assert(operation() != nullptr);
  assert(operation()->num_regions() == 1);
  assert(operation()->region(0).size() == 1);
  return operation()->region(0).front();
}

ModuleOp ModuleOp::Create(IrContext *context, Program *pointer) {
  ir::OpInfo info = context->GetRegisteredOpInfo(name());
  OperationArgument argument(info);
  argument.AddRegion()->emplace_back();
  argument.AddAttribute("program", PointerAttribute::get(context, pointer));
  return ModuleOp(Operation::Create(std::move(argument)));
}

void ModuleOp::Destroy() {
  if (operation()) {
    operation()->Destroy();
    *this = ModuleOp(nullptr);
  }
}

void ModuleOp::Verify() const {
  VLOG(4) << "Verifying inputs, outputs and attributes for: ModuleOp.";
  // Verify inputs:
  IR_ENFORCE(num_operands() == 0u, "The size of inputs must be equal to 0.");

  // Verify attributes:
  auto &attributes = this->attributes();
  auto iter = attributes.find("program");
  IR_ENFORCE(iter != attributes.end() && iter->second.isa<PointerAttribute>(),
             "Type of attribute: program is not right.");

  // Verify outputs:
  IR_ENFORCE(num_results() == 0u, "The size of inputs must be equal to 0.");
}

const char *GetParameterOp::attributes_name[attributes_num] = {
    "parameter_name"};

void GetParameterOp::Build(Builder &builder,
                           OperationArgument &argument,
                           const std::string &name,
                           Type type) {
  argument.attributes[attributes_name[0]] =
      ir::StrAttribute::get(builder.ir_context(), name);
  argument.output_types.emplace_back(type);
}

void GetParameterOp::Verify() const {
  VLOG(4) << "Verifying inputs, outputs and attributes for: GetParameterOp.";
  // Verify inputs:
  IR_ENFORCE(num_operands() == 0u, "The size of inputs must be equal to 0.");

  // Verify if attributes contain attribute name in attributes_name:
  auto &attributes = this->attributes();
  auto iter = attributes.find("parameter_name");
  IR_ENFORCE(iter != attributes.end() && iter->second.isa<StrAttribute>(),
             "Type of attribute: parameter_name is not right.");

  // Verify outputs type:
  IR_ENFORCE(num_results() == 1u, "The size of outputs must be equal to 1.");
}

const char *SetParameterOp::attributes_name[attributes_num] = {
    "parameter_name"};

void SetParameterOp::Build(Builder &builder,             // NOLINT
                           OperationArgument &argument,  // NOLINT
                           OpResult parameter,
                           const std::string &name) {
  argument.AddOperand(parameter);
  argument.AddAttribute(attributes_name[0],
                        ir::StrAttribute::get(builder.ir_context(), name));
}
void SetParameterOp::Verify() const {
  VLOG(4) << "Verifying inputs, outputs and attributes for: SetParameterOp.";
  // Verify inputs:
  IR_ENFORCE(num_operands() == 1, "The size of outputs must be equal to 1.");

  // Verify attributes:
  auto &attributes = this->attributes();
  auto iter = attributes.find("parameter_name");
  IR_ENFORCE(iter != attributes.end() && iter->second.isa<StrAttribute>(),
             "Type of attribute: parameter_name is not right.");

  // Verify outputs:
  IR_ENFORCE(num_results() == 0u, "The size of outputs must be equal to 0.");
}

void CombineOp::Build(Builder &builder,
                      OperationArgument &argument,
                      const std::vector<ir::OpResult> &inputs) {
  argument.inputs = inputs;
  std::vector<ir::Type> inputs_type(inputs.size());
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    inputs_type[idx] = inputs[idx].type();
  }
  argument.output_types.emplace_back(
      ir::VectorType::get(builder.ir_context(), inputs_type));
}

void CombineOp::Verify() const {
  // outputs.size() == 1
  IR_ENFORCE(num_results() == 1u, "The size of outputs must be equal to 1.");

  // output_type == Vector<Type>
  auto output_type = (*this)->result(0).type().dyn_cast<VectorType>();
  IR_ENFORCE(output_type,
             "The type of outputs[0] must be equal to VectorType.");

  // inputs.size() == outputs[0].size()
  auto input_num = num_operands();
  IR_ENFORCE(output_type.size() == input_num,
             "The size %d of output must be equal to size %d of inputs.",
             output_type.size(),
             input_num);

  // forall i in inputs.size(): inputs[i].type == outputs[0][i].type
  for (size_t i = 0; i < input_num; ++i) {
    auto type = (*this)->operand(i).type();
    IR_ENFORCE(output_type[i] == type,
               "The type %s of outputs[0][%d] must be "
               "equal to type %s of inputs[%d].",
               output_type[i],
               i,
               type,
               i);
  }
}

const char *SliceOp::attributes_name[attributes_num] = {"index"};
void SliceOp::Verify() const {
  // inputs.size() == 1
  auto input_size = num_operands();
  IR_ENFORCE(
      input_size == 1, "The size %d of inputs must be equal to 1.", input_size);

  // inputs[0].type == Vector<Type>
  auto input_type = (*this)->operand(0).type().dyn_cast<ir::VectorType>();
  IR_ENFORCE(input_type,
             "The type %s of inputs[0] must be equal to VectorType.",
             input_type);

  auto output_size = num_results();
  // outputs.size() == 1
  IR_ENFORCE(output_size == 1,
             "The size %d of outputs must be equal to 1.",
             output_size);

  // attributes contains index: Int32
  auto &attributes = this->attributes();
  IR_ENFORCE(attributes.count("index") != 0,
             "The attributes must contains index.");
  const ir::Attribute &attr = attributes.at("index");
  IR_ENFORCE(attr.isa<ir::Int32Attribute>(),
             "The attribute index must be INT32.");
  auto index = attr.dyn_cast<ir::Int32Attribute>().data();

  // index >= 0 and < inputs[0].size()
  IR_ENFORCE(
      index >= 0, "The index %d must be greater or equal than 0.", index);
  IR_ENFORCE(static_cast<size_t>(index) < input_type.size(),
             "The index %d must be less or equal than size %d of inputs[0].",
             index,
             input_type.size());

  // inputs[index].type == outputs[0].type
  auto output_type = (*this)->result(0).type();
  IR_ENFORCE(
      input_type[index] == output_type,
      "The type %s of inputs[%d] must be equal to type %s of outputs[0].",
      input_type[index],
      index,
      output_type);
}

const char *ConstantOp::attributes_name[attributes_num] = {"value"};

void ConstantOp::Build(Builder &builder,
                       OperationArgument &argument,
                       Attribute value,
                       Type output_type) {
  argument.AddAttribute("value", value);
  argument.output_types.push_back(output_type);
}

void ConstantOp::Verify() const {
  IR_ENFORCE(num_operands() == 0, "The size of inputs must be equal to 0.");
  IR_ENFORCE(num_results() == 1, "The size of outputs must be equal to 1.");
  IR_ENFORCE(attributes().count("value") > 0, "must has value attribute");
}

Attribute ConstantOp::value() const { return attributes().at("value"); }

}  // namespace ir

IR_DEFINE_EXPLICIT_TYPE_ID(ir::ModuleOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::GetParameterOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::SetParameterOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::CombineOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::SliceOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::ConstantLikeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::ConstantOp)
