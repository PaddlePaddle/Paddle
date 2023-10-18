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

#include "paddle/fluid/pir/drr/api/drr_pattern_context.h"

#include "paddle/fluid/pir/drr/pattern_graph.h"
#include "paddle/pir/core/enforce.h"

namespace pir {
namespace drr {

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
      new drr::Op(op_type, attributes, source_pattern_graph_.get())));
  return *owned_ops_.back();
}

const drr::Tensor& DrrPatternContext::SourceTensorPattern(
    const std::string& name) {
  return source_pattern_graph_->AddTensor(std::shared_ptr<drr::Tensor>(
      new drr::Tensor(name, source_pattern_graph_.get())));
}

const Op& DrrPatternContext::ResultOpPattern(
    const std::string& op_type,
    const std::unordered_map<std::string, Attribute>& attributes) {
  owned_ops_.push_back(std::shared_ptr<drr::Op>(
      new drr::Op(op_type, attributes, result_pattern_graph_.get())));
  return *owned_ops_.back();
}

drr::Tensor& DrrPatternContext::ResultTensorPattern(const std::string& name) {
  return result_pattern_graph_->AddTensor(std::shared_ptr<drr::Tensor>(
      new drr::Tensor(name, result_pattern_graph_.get())));
}

std::vector<Constraint> DrrPatternContext::constraints() const {
  return constraints_;
}

// void DrrPatternContext::RequireEqual(const Attribute& first, const Attribute&
// second) {
//   auto constrain_fn = [&](const MatchContext& match_context) {
//     return match_context.Attr(first.id()) == match_context.Attr(second.id());
//   };
//   constraints_.emplace_back(constrain_fn);
// }

void DrrPatternContext::RequireEqual(const TensorShape& first,
                                     const TensorShape& second) {
  // Note: we capture the datas by value for constrain_fn
  // because the datas are destructed before running constrain_fn.
  auto constrain_fn = [=](const MatchContext& match_context) {
    return match_context.Tensor(first.tensor_name()).Shape() ==
           match_context.Tensor(second.tensor_name()).Shape();
  };
  constraints_.emplace_back(constrain_fn);
}

void DrrPatternContext::RequireEqual(const TensorDataType& first,
                                     const TensorDataType& second) {
  // Note: we capture the datas by value for constrain_fn
  // because the datas are destructed before running constrain_fn.
  auto constrain_fn = [=](const MatchContext& match_context) {
    return match_context.Tensor(first.tensor_name()).Dtype() ==
           match_context.Tensor(second.tensor_name()).Dtype();
  };
  constraints_.emplace_back(constrain_fn);
}

void DrrPatternContext::RequireNativeCall(
    const std::function<bool(const MatchContext&)>& custom_fn) {
  constraints_.emplace_back(custom_fn);
}

void Op::operator()(const Tensor& arg, const Tensor* out) const {
  std::vector<const Tensor*> inputs{&arg};
  std::vector<const Tensor*> outputs{out};
  pattern_graph_->AddOpCall(std::make_shared<OpCall>(this, inputs, outputs));
}

void Op::operator()(const std::vector<const Tensor*>& args,
                    const std::vector<const Tensor*>& outputs) const {
  pattern_graph_->AddOpCall(std::make_shared<OpCall>(this, args, outputs));
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

const char Tensor::NONE_TENSOR_NAME[] = "__@none_tensor@__";

void Tensor::Assign(const Tensor& other) {
  dynamic_cast<ResultPatternGraph*>(pattern_graph_)->AssignTensor(*this, other);
}

void Tensor::operator=(const Tensor& other) const {  // NOLINT
  // The two tensor must be in the same pattern graph.
  IR_ENFORCE(this->pattern_graph_ == other.pattern_graph_);
  if (other.name_.find(Op::prefix) == 0 &&
      name_.find(Op::prefix) == std::string::npos) {
    other.pattern_graph_->UpdateTmpTensor(other.name_, this->name_);
  }
}

}  // namespace drr
}  // namespace pir
