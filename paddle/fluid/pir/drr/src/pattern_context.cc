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

#include <memory>

#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/fluid/pir/drr/src/pattern_graph.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/phi/common/data_type.h"

namespace paddle::drr {

DrrPatternContext::DrrPatternContext() {
  source_pattern_graph_ = std::make_shared<SourcePatternGraph>();
  result_pattern_graph_ = std::make_shared<ResultPatternGraph>();
}

drr::SourcePattern DrrPatternContext::SourcePattern() {
  return drr::SourcePattern(this);
}

const Op& DrrPatternContext::SourceOpPattern(
    const std::string& op_type,
    const std::unordered_map<std::string, Attribute>& attributes) {
  owned_ops_.push_back(std::shared_ptr<drr::Op>(
      new drr::Op(op_type, source_pattern_graph_.get(), attributes)));
  return *owned_ops_.back();
}

drr::Tensor& DrrPatternContext::SourceTensorPattern(const std::string& name) {
  return source_pattern_graph_->AddTensor(std::shared_ptr<drr::Tensor>(
      new drr::Tensor(name, source_pattern_graph_.get())));
}

const Op& DrrPatternContext::ResultOpPattern(
    const std::string& op_type,
    const std::unordered_map<std::string, Attribute>& attributes,
    const std::unordered_map<std::string, Attribute>& runtime_attributes) {
  owned_ops_.push_back(std::shared_ptr<drr::Op>(new drr::Op(
      op_type, result_pattern_graph_.get(), attributes, runtime_attributes)));
  return *owned_ops_.back();
}

drr::Tensor& DrrPatternContext::ResultTensorPattern(const std::string& name) {
  return result_pattern_graph_->AddTensor(std::shared_ptr<drr::Tensor>(
      new drr::Tensor(name, result_pattern_graph_.get())));
}

std::vector<Constraint> DrrPatternContext::constraints() const {
  return constraints_;
}

void DrrPatternContext::AddConstraint(const ConstraintFunction& constraint_fn) {
  constraints_.emplace_back(constraint_fn);
}

std::vector<PostProcess> DrrPatternContext::post_processes() const {
  return post_processes_;
}

void DrrPatternContext::AddPostProcess(
    const PostProcessFunction& post_process_fn) {
  post_processes_.emplace_back(post_process_fn);
}

void Op::operator()(const Tensor& arg, const Tensor* out) const {
  std::vector<const Tensor*> inputs{&arg};
  std::vector<const Tensor*> outputs{out};
  pattern_graph_->AddOpCall(std::make_shared<OpCall>(this, inputs, outputs));
}

void Op::operator()(const std::vector<const Tensor*>& args,
                    const std::vector<const Tensor*>& outputs) const {
  std::vector<const Tensor*> inputs_ptr;
  std::vector<const Tensor*> outputs_ptr;
  for (auto arg : args) {
    const Tensor* lower_input_ptr =
        (pattern_graph_->id2owned_tensor().at(arg->name())).get();
    inputs_ptr.emplace_back(lower_input_ptr);
  }
  for (auto output : outputs) {
    const Tensor* lower_output_ptr =
        (pattern_graph_->id2owned_tensor().at(output->name())).get();
    outputs_ptr.emplace_back(lower_output_ptr);
  }
  pattern_graph_->AddOpCall(
      std::make_shared<OpCall>(this, inputs_ptr, outputs_ptr));
}

Tensor& Op::operator()(const Tensor& arg) const {
  std::vector<const Tensor*> inputs{&arg};
  auto& out = pattern_graph_->AddTmpTensor(std::shared_ptr<Tensor>(new Tensor(
      prefix + op_type_name_ + "_" + std::to_string(count++), pattern_graph_)));
  std::vector<const Tensor*> outputs{&out};
  pattern_graph_->AddOpCall(std::make_shared<OpCall>(this, inputs, outputs));
  return out;
}

Tensor& Op::operator()(const Tensor& arg1, const Tensor& arg2) const {
  std::vector<const Tensor*> inputs{&arg1, &arg2};
  auto& out = pattern_graph_->AddTmpTensor(std::shared_ptr<Tensor>(new Tensor(
      prefix + op_type_name_ + "_" + std::to_string(count++), pattern_graph_)));
  std::vector<const Tensor*> outputs{&out};
  pattern_graph_->AddOpCall(std::make_shared<OpCall>(this, inputs, outputs));
  return out;
}

Tensor& Op::operator()(const Tensor& arg0,
                       const Tensor& arg1,
                       const Tensor& arg2) const {
  std::vector<const Tensor*> inputs{&arg0, &arg1, &arg2};
  auto& out = pattern_graph_->AddTmpTensor(std::shared_ptr<Tensor>(new Tensor(
      prefix + op_type_name_ + "_" + std::to_string(count++), pattern_graph_)));
  std::vector<const Tensor*> outputs{&out};
  pattern_graph_->AddOpCall(std::make_shared<OpCall>(this, inputs, outputs));
  return out;
}

Tensor& Op::operator()() const {
  std::vector<const Tensor*> inputs{};
  auto& out = pattern_graph_->AddTmpTensor(std::shared_ptr<Tensor>(new Tensor(
      prefix + op_type_name_ + "_" + std::to_string(count++), pattern_graph_)));
  std::vector<const Tensor*> outputs{&out};
  pattern_graph_->AddOpCall(std::make_shared<OpCall>(this, inputs, outputs));
  return out;
}

thread_local int64_t Op::count = 0;
const char* Op::prefix = "@drr_temp@_";

void Tensor::Assign(const Tensor& other) {
  dynamic_cast<ResultPatternGraph*>(pattern_graph_)->AssignTensor(*this, other);
}

void Tensor::operator=(const Tensor& other) const {  // NOLINT
  // The two tensor must be in the same pattern graph.
  PADDLE_ENFORCE_EQ(
      this->pattern_graph_,
      other.pattern_graph_,
      common::errors::InvalidArgument("Matching failed."
                                      "Two Tensors must be in the same pattern "
                                      "graph to make the '=' judgment."));
  if (other.name_.find(Op::prefix) == 0 &&
      name_.find(Op::prefix) == std::string::npos) {
    other.pattern_graph_->UpdateTmpTensor(other.name_, this->name_);
  }
}

const drr::Op& ResultPattern::Op(
    const std::string& op_type,
    const std::unordered_map<std::string, Attribute>& attributes,
    const std::unordered_map<std::string, Attribute>& runtime_attributes) {
  return ctx_->ResultOpPattern(op_type, attributes, runtime_attributes);
}

drr::Tensor& ResultPattern::Tensor(const std::string& name) {
  return ctx_->ResultTensorPattern(name);
}

drr::Tensor& ResultPattern::InputNoneTensor() {
  return ctx_->ResultTensorPattern(Tensor::RESULT_INPUT_NONE_TENSOR_NAME);
}

drr::Tensor& ResultPattern::OutputNoneTensor() {
  return ctx_->ResultTensorPattern(Tensor::RESULT_OUTPUT_NONE_TENSOR_NAME);
}

Attribute ResultPattern::StrAttr(const std::string& value) const {
  return ComputeAttr(
      [=](const MatchContext& match_ctx) -> std::string { return value; });
}

Attribute ResultPattern::BoolAttr(bool value) const {
  return ComputeAttr(
      [=](const MatchContext& match_ctx) -> bool { return value; });
}

Attribute ResultPattern::Int32Attr(int32_t value) const {
  return ComputeAttr(
      [=](const MatchContext& match_ctx) -> int32_t { return value; });
}

Attribute ResultPattern::Int64Attr(int64_t value) const {
  return ComputeAttr(
      [=](const MatchContext& match_ctx) -> int64_t { return value; });
}

Attribute ResultPattern::Float32Attr(float value) const {
  return ComputeAttr(
      [=](const MatchContext& match_ctx) -> float { return value; });
}

Attribute ResultPattern::VectorInt64Attr(
    const std::vector<int64_t>& value) const {
  return ComputeAttr(
      [=](const MatchContext& match_ctx) -> std::vector<int64_t> {
        return value;
      });
}

Attribute ResultPattern::VectorInt32Attr(
    const std::vector<int32_t>& value) const {
  return ComputeAttr(
      [=](const MatchContext& match_ctx) -> std::vector<int32_t> {
        return value;
      });
}

Attribute ResultPattern::VectorFloatAttr(
    const std::vector<float>& value) const {
  return ComputeAttr([=](const MatchContext& match_ctx) -> std::vector<float> {
    return value;
  });
}

Attribute ResultPattern::DataTypeAttr(const std::string& value) const {
  return ComputeAttr([=](const MatchContext& match_ctx) -> phi::DataType {
    PADDLE_ENFORCE_EQ(dialect::StringToDataTypeMap().count(value) > 0,
                      true,
                      common::errors::InvalidArgument(
                          "The DataTypeAttr %s is not supported.", value));
    return dialect::StringToDataTypeMap().at(value);
  });
}

Attribute ResultPattern::PlaceAttr(const std::string& value) const {
  return ComputeAttr([=](const MatchContext& match_ctx) -> phi::Place {
    PADDLE_ENFORCE_EQ(dialect::StringToPlaceMap().count(value) > 0,
                      true,
                      common::errors::InvalidArgument(
                          "The PlaceAttr %s is not supported.", value));
    return dialect::StringToPlaceMap().at(value);
  });
}

Attribute ResultPattern::DataLayoutAttr(const std::string& value) const {
  return ComputeAttr([=](const MatchContext& match_ctx) -> phi::DataLayout {
    PADDLE_ENFORCE_EQ(dialect::StringToDataLayoutMap().count(value) > 0,
                      true,
                      common::errors::InvalidArgument(
                          "The DataLayoutAttr %s is not supported.", value));
    return dialect::StringToDataLayoutMap().at(value);
  });
}

Attribute ResultPattern::ComputeAttr(
    const AttrComputeFunc& attr_compute_func) const {
  return ComputeAttribute(attr_compute_func);
}

drr::ResultPattern SourcePattern::ResultPattern() const {
  return drr::ResultPattern(ctx_);
}

const drr::Op& SourcePattern::Op(
    const std::string& op_type,
    const std::unordered_map<std::string, Attribute>& attributes) {
  return ctx_->SourceOpPattern(op_type, attributes);
}

const drr::Tensor& SourcePattern::Tensor(const std::string& name) {
  return ctx_->SourceTensorPattern(name);
}

Attribute SourcePattern::Attr(const std::string& attr_name) const {
  return NormalAttribute(attr_name);
}

void SourcePattern::AddConstraint(const ConstraintFunction& constraint_fn) {
  ctx_->AddConstraint(constraint_fn);
}

void SourcePattern::AddPostProcess(const PostProcessFunction& post_process_fn) {
  ctx_->AddPostProcess(post_process_fn);
}

drr::Tensor& SourcePattern::InputNoneTensor() {
  return ctx_->SourceTensorPattern(Tensor::SOURCE_INPUT_NONE_TENSOR_NAME);
}

drr::Tensor& SourcePattern::OutputNoneTensor() {
  return ctx_->SourceTensorPattern(Tensor::SOURCE_OUTPUT_NONE_TENSOR_NAME);
}

}  // namespace paddle::drr
