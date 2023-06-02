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
#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/phi/core/enforce.h"

namespace ir {

const char *ModuleOp::attributes_name[attributes_num] = {"program"};

Program *ModuleOp::program() {
  const AttributeMap &attr = operation()->attributes();
  auto iter = attr.find("program");
  if (iter == attr.end() || !iter->second) return nullptr;
  return static_cast<Program *>(
      iter->second.dyn_cast<PointerAttribute>().data());
}

Block *ModuleOp::block() {
  assert(operation() != nullptr);
  assert(operation()->num_regions() == 1);
  assert(operation()->GetRegion(0).size() == 1);
  return operation()->GetRegion(0).front();
}

ModuleOp ModuleOp::create(IrContext *context, Program *pointer) {
  ir::OpInfo info = context->GetRegisteredOpInfo(name());
  OperationArgument argument(info);
  argument.AddRegion()->emplace_back();
  argument.AddAttribute("program", PointerAttribute::get(context, pointer));
  return ModuleOp(Operation::create(std::move(argument)));
}

void ModuleOp::destroy() {
  if (operation()) {
    operation()->destroy();
    *this = ModuleOp(nullptr);
  }
}

void ModuleOp::verify(const std::vector<ir::OpResult> &inputs,
                      const std::vector<ir::Type> &outputs,
                      const ir::AttributeMap &attributes) {
  VLOG(4) << "Verifying inputs, outputs and attributes for: ModuleOp.";
  // Verify inputs type:
  if (inputs.size() != 0) {
    throw("The size of inputs must be equal to 0.");
  }

  // Verify if attributes contain attribute name in attributes_name:
  auto iter = attributes.find("program");
  if (iter == attributes.end() || !iter->second.isa<PointerAttribute>()) {
    throw("Type of attribute: program is not right.");
  }

  // Verify outputs type:
  if (outputs.size() != 0) {
    throw("The size of outputs must be equal to 0.");
  }
}

const char *GetParameterOp::attributes_name[attributes_num] = {
    "parameter_name"};

void GetParameterOp::verify(const std::vector<ir::OpResult> &inputs,
                            const std::vector<ir::Type> &outputs,
                            const ir::AttributeMap &attributes) {
  VLOG(4) << "Verifying inputs, outputs and attributes for: GetParameterOp.";
  // Verify inputs type:
  if (inputs.size() != 0) {
    throw("The size of inputs must be equal to 0.");
  }
  // Verify outputs type:
  if (outputs.size() != 1) {
    throw("The size of outputs must be equal to 1.");
  }
  // Verify if attributes contain attribute name in attributes_name:
  if (!attributes.at("parameter_name").isa<StrAttribute>()) {
    throw("Type of attribute: parameter_name is not right.");
  }
}

const char *SetParameterOp::attributes_name[attributes_num] = {
    "parameter_name"};

void SetParameterOp::verify(const std::vector<ir::OpResult> &inputs,
                            const std::vector<ir::Type> &outputs,
                            const ir::AttributeMap &attributes) {
  VLOG(4) << "Verifying inputs, outputs and attributes for: SetParameterOp.";
  // Verify inputs type:
  if (inputs.size() != 1) {
    throw("The size of inputs must be equal to 1.");
  }
  // Verify outputs type:
  if (outputs.size() != 0) {
    throw("The size of outputs must be equal to 0.");
  }
  // Verify if attributes contain attribute name in attributes_name:
  if (!attributes.at("parameter_name").isa<StrAttribute>()) {
    throw("Type of attribute: parameter_name is not right.");
  }
}

void CombineOp::verify(const std::vector<ir::OpResult> &inputs,
                       const std::vector<ir::Type> &outputs,
                       const ir::AttributeMap &attributes) {
  // outputs.size() == 1
  PADDLE_ENFORCE_EQ(
      outputs.size(),
      1,
      phi::errors::PreconditionNotMet(
          "The size %d of outputs must be equal to 1.", outputs.size()));
  // outputs[0].type == Vector<Type>
  PADDLE_ENFORCE(outputs[0].isa<ir::VectorType>(),
                 phi::errors::PreconditionNotMet(
                     "The type %s of outputs[0] must be equal to VectorType.",
                     outputs[0]));
  ir::VectorType output_type = outputs[0].dyn_cast<ir::VectorType>();
  // inputs.size() == outputs[0].size()
  PADDLE_ENFORCE_EQ(
      output_type.size(),
      inputs.size(),
      phi::errors::PreconditionNotMet(
          "The size %d of outputs[0] must be equal to size %d of inputs.",
          output_type.size(),
          inputs.size()));

  // forall i in inputs.size(): inputs[i].type == outputs[0][i].type
  for (size_t i = 0; i < inputs.size(); i++) {
    PADDLE_ENFORCE_EQ(
        output_type[i],
        inputs[i].type(),
        phi::errors::PreconditionNotMet("The type %s of outputs[0][%d] must be "
                                        "equal to type %s of inputs[%d].",
                                        output_type[i],
                                        i,
                                        inputs[i].type(),
                                        i));
  }
}

const char *SliceOp::attributes_name[attributes_num] = {"index"};
void SliceOp::verify(const std::vector<ir::OpResult> &inputs,
                     const std::vector<ir::Type> &outputs,
                     const ir::AttributeMap &attributes) {
  // inputs.size() == 1
  PADDLE_ENFORCE_EQ(
      inputs.size(),
      1,
      phi::errors::PreconditionNotMet(
          "The size %d of inputs must be equal to 1.", inputs.size()));

  // inputs[0].type == Vector<Type>
  PADDLE_ENFORCE(inputs[0].type().isa<ir::VectorType>(),
                 phi::errors::PreconditionNotMet(
                     "The type %s of inputs[0] must be equal to VectorType.",
                     inputs[0].type()));
  ir::VectorType input_type = inputs[0].type().dyn_cast<ir::VectorType>();

  // outputs.size() == 1
  PADDLE_ENFORCE_EQ(
      outputs.size(),
      1,
      phi::errors::PreconditionNotMet(
          "The size %d of outputs must be equal to 1.", outputs.size()));

  // attributes contains index: Int32
  PADDLE_ENFORCE_NE(
      attributes.count("index"),
      0,
      phi::errors::PreconditionNotMet("The attributes must contains index."));
  const ir::Attribute &attr = attributes.at("index");
  PADDLE_ENFORCE(
      attr.isa<ir::Int32_tAttribute>(),
      phi::errors::PreconditionNotMet("The attribute index must be INT32."));
  auto index = attr.dyn_cast<ir::Int32_tAttribute>().data();

  // index >= 0 and < inputs[0].size()
  PADDLE_ENFORCE_GE(
      index,
      0,
      phi::errors::PreconditionNotMet(
          "The index %d must be greater or equal than 0.", index));
  PADDLE_ENFORCE_LT(
      index,
      input_type.size(),
      phi::errors::PreconditionNotMet(
          "The index %d must be less or equal than size %d of inputs[0].",
          index,
          input_type.size()));

  // inputs[index].type == outputs[0].type
  PADDLE_ENFORCE_EQ(
      input_type[index],
      outputs[0],
      phi::errors::PreconditionNotMet(
          "The type %s of inputs[%d] must be equal to type %s of outputs[0].",
          input_type[index],
          index,
          outputs[0]));
}

const char *ConstantOp::attributes_name[attributes_num] = {"value"};

void ConstantOp::build(Builder &builder,
                       OperationArgument &argument,
                       Attribute value,
                       Type output_type) {
  argument.AddAttribute("value", value);
  argument.output_types.push_back(output_type);
}

void ConstantOp::verify(const std::vector<ir::OpResult> &inputs,
                        const std::vector<ir::Type> &outputs,
                        const ir::AttributeMap &attributes) {
  IR_ENFORCE(inputs.size() == 0, "The size of inputs must be equal to 0.");
  IR_ENFORCE(outputs.size() == 1, "The size of outputs must be equal to 1.");
  IR_ENFORCE(attributes.count("value") > 0,
             "Type of attribute: value is not right.");
}

Attribute ConstantOp::value() { return operation()->attributes().at("value"); }

}  // namespace ir
