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

#include "paddle/pir/core/builtin_op.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_type.h"

namespace pir {

const char *ModuleOp::attributes_name[attributes_num] = {"program"};  // NOLINT

void PassStopGradientsDefaultly(OperationArgument &argument) {  // NOLINT
  VLOG(10) << "Builder construction stop gradient for OpResults.";
  bool stop_gradient = true;
  for (auto value : argument.inputs) {
    auto attr = value.attribute<BoolAttribute>(kStopGradientAttrName);
    if (attr && !attr.data()) {
      stop_gradient = false;
      break;
    }
  }
  std::vector<pir::Attribute> outs_stop_gradient(
      argument.output_types.size(),
      pir::BoolAttribute::get(pir::IrContext::Instance(), stop_gradient));
  argument.AddAttribute(
      kStopGradientAttrName,
      pir::ArrayAttribute::get(pir::IrContext::Instance(), outs_stop_gradient));
}

void RefreshStopGradientsDefaultly(Operation *op) {
  bool stop_gradient = true;
  for (auto value : op->operands_source()) {
    auto attr = value.attribute<BoolAttribute>(kStopGradientAttrName);
    if (attr && !attr.data()) {
      stop_gradient = false;
      break;
    }
  }
  std::vector<pir::Attribute> outs_stop_gradient(
      op->results().size(),
      pir::BoolAttribute::get(pir::IrContext::Instance(), stop_gradient));
  op->set_attribute(
      kStopGradientAttrName,
      pir::ArrayAttribute::get(pir::IrContext::Instance(), outs_stop_gradient));
}
Program *ModuleOp::program() {
  const AttributeMap &attr = this->attributes();
  auto iter = attr.find("program");
  if (iter == attr.end() || !iter->second) return nullptr;
  return static_cast<Program *>(
      iter->second.dyn_cast<PointerAttribute>().data());
}

Block &ModuleOp::block() {
  IR_ENFORCE(operation()->num_regions(),
             "The region size of ModuleOp must be equal to 1.");
  auto &region = (*this)->region(0);
  IR_ENFORCE(region.size() == 1,
             "The region size of ModuleOp must be equal to 1.");
  return region.front();
}

ModuleOp ModuleOp::Create(IrContext *context, Program *pointer) {
  pir::OpInfo info = context->GetRegisteredOpInfo(name());
  OperationArgument argument(info);
  argument.AddRegion(nullptr);
  argument.AddAttribute("program", PointerAttribute::get(context, pointer));
  Operation *op = Operation::Create(std::move(argument));
  op->region(0).emplace_back();
  return ModuleOp(op);
}

void ModuleOp::Destroy() {
  if (operation()) {
    operation()->Destroy();
    *this = ModuleOp(nullptr);
  }
}

void ModuleOp::VerifySig() const {
  VLOG(10) << "Verifying inputs, outputs and attributes for: ModuleOp.";
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

const char *ParameterOp::attributes_name[attributes_num] = {  // NOLINT
    "parameter_name"};

void ParameterOp::Build(Builder &builder,
                        OperationArgument &argument,
                        const std::string &name,
                        Type type) {
  argument.attributes[attributes_name[0]] =
      pir::StrAttribute::get(builder.ir_context(), name);
  argument.output_types.emplace_back(type);
  PassStopGradients(argument);
}

void ParameterOp::PassStopGradients(OperationArgument &argument) {
  std::vector<pir::Attribute> outs_stop_gradient(
      1, pir::BoolAttribute::get(pir::IrContext::Instance(), false));
  argument.AddAttribute(
      kStopGradientAttrName,
      pir::ArrayAttribute::get(pir::IrContext::Instance(), outs_stop_gradient));
}

std::string ParameterOp::param_name() const {
  return attribute<StrAttribute>("parameter_name").AsString();
}
void ParameterOp::VerifySig() const {
  VLOG(10) << "Verifying inputs, outputs and attributes for: ParameterOp.";
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

const char *SetParameterOp::attributes_name[attributes_num] = {  // NOLINT
    "parameter_name"};

void SetParameterOp::Build(Builder &builder,             // NOLINT
                           OperationArgument &argument,  // NOLINT
                           Value parameter,
                           const std::string &name) {
  argument.AddInput(parameter);
  argument.AddAttribute(attributes_name[0],
                        pir::StrAttribute::get(builder.ir_context(), name));
}
void SetParameterOp::VerifySig() const {
  VLOG(10) << "Verifying inputs, outputs and attributes for: SetParameterOp.";
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

const char *ShadowOutputOp::attributes_name[attributes_num] = {  // NOLINT
    "output_name"};

void ShadowOutputOp::Build(Builder &builder,             // NOLINT
                           OperationArgument &argument,  // NOLINT
                           Value parameter,
                           const std::string &name) {
  argument.AddInput(parameter);
  argument.AddAttribute(attributes_name[0],
                        pir::StrAttribute::get(builder.ir_context(), name));
}
void ShadowOutputOp::VerifySig() const {
  VLOG(10) << "Verifying inputs, outputs and attributes for: ShadowOutputOp.";
  // Verify inputs:
  IR_ENFORCE(num_operands() == 1, "The size of outputs must be equal to 1.");

  // Verify attributes:
  auto &attributes = this->attributes();
  auto iter = attributes.find("output_name");
  IR_ENFORCE(iter != attributes.end() && iter->second.isa<StrAttribute>(),
             "Type of attribute: output_name is not right.");

  // Verify outputs:
  IR_ENFORCE(num_results() == 0u, "The size of outputs must be equal to 0.");
}

void CombineOp::Build(Builder &builder,
                      OperationArgument &argument,
                      const std::vector<Value> &inputs) {
  argument.inputs = inputs;
  std::vector<pir::Type> inputs_type(inputs.size());
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    inputs_type[idx] = inputs[idx].type();
  }
  argument.output_types.emplace_back(
      pir::VectorType::get(builder.ir_context(), inputs_type));
  PassStopGradientsDefaultly(argument);
}

void CombineOp::VerifySig() const {
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

const char *SliceOp::attributes_name[attributes_num] = {"index"};  // NOLINT

void SliceOp::Build(Builder &builder,
                    OperationArgument &argument,
                    Value input,
                    int index) {
  argument.inputs = {input};
  argument.output_types.emplace_back(input.type()
                                         .dyn_cast<pir::VectorType>()
                                         .data()[static_cast<size_t>(index)]);
  PassStopGradients(argument, index);

  argument.AddAttribute(
      "index", pir::Int32Attribute::get(pir::IrContext::Instance(), index));
}

void SliceOp::PassStopGradients(OperationArgument &argument, int index) {
  std::vector<pir::Attribute> outs_stop_gradient(
      1, pir::BoolAttribute::get(pir::IrContext::Instance(), true));
  if (auto input = argument.inputs[0].dyn_cast<pir::OpResult>()) {
    auto *defining_op = input.owner();
    if (defining_op && defining_op->isa<CombineOp>()) {
      IR_ENFORCE(defining_op->HasAttribute(kStopGradientAttrName),
                 "Required CombineOp must have attribute %s",
                 kStopGradientAttrName);
      auto attrs = defining_op->attribute(kStopGradientAttrName)
                       .dyn_cast<pir::ArrayAttribute>()
                       .AsVector();
      outs_stop_gradient[0] = attrs[index];
    }
  }
  argument.AddAttribute(
      kStopGradientAttrName,
      pir::ArrayAttribute::get(pir::IrContext::Instance(), outs_stop_gradient));
}

void SliceOp::RefreshStopGradients() {
  std::vector<pir::Attribute> outs_stop_gradient(
      1, pir::BoolAttribute::get(pir::IrContext::Instance(), true));
  auto index = attribute("index").dyn_cast<pir::Int32Attribute>().data();
  if (auto input = (*this)->operand_source(0).dyn_cast<pir::OpResult>()) {
    auto *defining_op = input.owner();
    if (defining_op && defining_op->isa<CombineOp>()) {
      IR_ENFORCE(defining_op->HasAttribute(kStopGradientAttrName),
                 "Required CombineOp must have attribute %s",
                 kStopGradientAttrName);
      auto attr = defining_op->attribute(kStopGradientAttrName)
                      .dyn_cast<pir::ArrayAttribute>();
      outs_stop_gradient[0] = attr.at(static_cast<size_t>(index));
    }
  }
  (*this)->set_attribute(
      kStopGradientAttrName,
      pir::ArrayAttribute::get(pir::IrContext::Instance(), outs_stop_gradient));
}

void SliceOp::VerifySig() const {
  // inputs.size() == 1
  auto input_size = num_operands();
  IR_ENFORCE(
      input_size == 1, "The size %d of inputs must be equal to 1.", input_size);

  // inputs[0].type == Vector<Type>
  auto input_type = (*this)->operand(0).type().dyn_cast<pir::VectorType>();
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
  const pir::Attribute &attr = attributes.at("index");
  IR_ENFORCE(attr.isa<pir::Int32Attribute>(),
             "The attribute index must be INT32.");
  auto index = attr.dyn_cast<pir::Int32Attribute>().data();

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

void SplitOp::Build(Builder &builder,
                    OperationArgument &argument,
                    Value input) {
  argument.inputs = {input};
  for (size_t idx = 0; idx < input.type().dyn_cast<pir::VectorType>().size();
       ++idx) {
    argument.output_types.emplace_back(
        input.type().dyn_cast<pir::VectorType>().data()[idx]);
  }

  PassStopGradients(argument);
}

void SplitOp::PassStopGradients(OperationArgument &argument) {
  std::vector<bool> defaut_stop_gradients(argument.output_types.size(), true);
  if (auto input = argument.inputs[0].dyn_cast<OpResult>()) {
    auto *defining_op = input.owner();
    if (defining_op && defining_op->isa<CombineOp>()) {
      IR_ENFORCE(argument.output_types.size(),
                 defining_op->num_operands(),
                 "Required SplitOp.output.size() == CombineOp.input.size(), "
                 "but received %d != %d",
                 argument.output_types.size(),
                 defining_op->num_operands());
      for (uint32_t i = 0; i < defining_op->num_operands(); ++i) {
        auto attr =
            defining_op->operand_source(i).attribute<pir::BoolAttribute>(
                kStopGradientAttrName);
        if (attr) {
          defaut_stop_gradients[i] = attr.data();
        }
      }
    } else if (defining_op &&
               defining_op->HasAttribute(kStopGradientAttrName)) {
      bool stop_gradient = defining_op->attribute(kStopGradientAttrName)
                               .dyn_cast<pir::ArrayAttribute>()
                               .at(0)
                               .dyn_cast<pir::BoolAttribute>()
                               .data();
      defaut_stop_gradients.assign(defaut_stop_gradients.size(), stop_gradient);
    }
  }

  std::vector<pir::Attribute> outs_stop_gradient;
  outs_stop_gradient.reserve(argument.output_types.size());
  for (auto stop_gradient : defaut_stop_gradients) {
    outs_stop_gradient.push_back(
        pir::BoolAttribute::get(pir::IrContext::Instance(), stop_gradient));
  }
  argument.AddAttribute(
      kStopGradientAttrName,
      pir::ArrayAttribute::get(pir::IrContext::Instance(), outs_stop_gradient));
}

void SplitOp::RefreshStopGradients() {
  std::vector<bool> default_stop_gradients((*this)->num_results(), true);
  if (auto input = (*this)->operand_source(0).dyn_cast<OpResult>()) {
    auto *defining_op = input.owner();
    if (defining_op && defining_op->isa<CombineOp>()) {
      IR_ENFORCE((*this)->num_results(),
                 defining_op->num_operands(),
                 "Required SplitOp.output.size() == CombineOp.input.size(), "
                 "but received %d != %d",
                 (*this)->num_results(),
                 defining_op->num_operands());
      for (uint32_t i = 0; i < defining_op->num_operands(); ++i) {
        auto value = defining_op->operand_source(i);
        if (!value) continue;
        auto *operand_defining_op = value.dyn_cast<OpResult>().owner();
        if (operand_defining_op->HasAttribute(kStopGradientAttrName)) {
          auto attrs = operand_defining_op->attribute(kStopGradientAttrName)
                           .dyn_cast<pir::ArrayAttribute>()
                           .AsVector();
          default_stop_gradients[i] = attrs[value.dyn_cast<OpResult>().index()]
                                          .dyn_cast<pir::BoolAttribute>()
                                          .data();
        }
      }
    } else if (defining_op &&
               defining_op->HasAttribute(kStopGradientAttrName)) {
      bool stop_gradient = defining_op->attribute(kStopGradientAttrName)
                               .dyn_cast<pir::ArrayAttribute>()
                               .AsVector()[0]
                               .dyn_cast<pir::BoolAttribute>()
                               .data();
      default_stop_gradients.assign(default_stop_gradients.size(),
                                    stop_gradient);
    }
  }

  std::vector<pir::Attribute> outs_stop_gradient;
  outs_stop_gradient.reserve(num_results());
  for (auto stop_gradient : default_stop_gradients) {
    outs_stop_gradient.push_back(
        pir::BoolAttribute::get(pir::IrContext::Instance(), stop_gradient));
  }
  (*this)->set_attribute(
      kStopGradientAttrName,
      pir::ArrayAttribute::get(pir::IrContext::Instance(), outs_stop_gradient));
}

void SplitOp::VerifySig() const {
  // inputs.size() == 1
  IR_ENFORCE(num_operands() == 1u, "The size of inputs must be equal to 1.");

  // input_type == Vector<Type>
  auto input_type = (*this)->operand(0).type().dyn_cast<VectorType>();
  IR_ENFORCE(input_type, "The type of inputs[0] must be equal to VectorType.");

  // inputs[0].size() == outputs.size()
  auto output_num = num_results();
  IR_ENFORCE(input_type.size() == output_num,
             "The size %d of output must be equal to size %d of inputs.",
             output_num,
             input_type.size());

  // for all i in outputs.size(): outputs[i].type == inputs[0][i].type
  // TODO(@xiongkun) consult zhangbo to check what to do with null type.
}

const char *ConstantOp::attributes_name[attributes_num] = {"value"};  // NOLINT

void ConstantOp::Build(Builder &builder,
                       OperationArgument &argument,
                       Attribute value,
                       Type output_type) {
  argument.AddAttribute("value", value);
  argument.output_types.push_back(output_type);
}

void ConstantOp::VerifySig() const {
  IR_ENFORCE(num_operands() == 0, "The size of inputs must be equal to 0.");
  IR_ENFORCE(num_results() == 1, "The size of outputs must be equal to 1.");
  IR_ENFORCE(attributes().count("value") > 0, "must has value attribute");
}

Attribute ConstantOp::value() const { return attributes().at("value"); }

void ConstantTensorOp::VerifySig() const {
  ConstantOp::VerifySig();
  IR_ENFORCE(value().isa<pir::TensorNameAttribute>(),
             "Type of value must be strattribute");
}

ConstantTensorOp ConstantTensorOp::dyn_cast(Operation *op) {
  if (ConstantTensorOp::classof(op)) return ConstantTensorOp(op);
  return ConstantTensorOp(nullptr);
}

bool ConstantTensorOp::classof(const Operation *op) {
  return ConstantOp::classof(op) && op &&
         op->attribute("value").isa<TensorNameAttribute>();
}

std::string ConstantTensorOp::tensor_name() {
  return value().dyn_cast<pir::TensorNameAttribute>().data();
}
}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::ModuleOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::ParameterOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::SetParameterOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::ShadowOutputOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::CombineOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::SliceOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::SplitOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::ConstantLikeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::ConstantOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::ConstantTensorOp)
