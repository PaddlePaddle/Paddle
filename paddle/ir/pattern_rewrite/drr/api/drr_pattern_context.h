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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/ir/pattern_rewrite/drr/api/match_context.h"

namespace ir {
namespace drr {

class Op;
class Tensor;
class OpCall;
class SourcePattern;
class ResultPattern;
class PatternGraph;
class SourcePatternGraph;
class ResultPatternGraph;

class Attribute {
 public:
  explicit Attribute(const std::string& id) : attr_id_(id) {}

  const std::string& id() const { return attr_id_; }

 private:
  std::string attr_id_;
};

class TensorShape {
 public:
  explicit TensorShape(const std::string& name) : name_(name) {}

  const std::string& name() const { return name_; }

 private:
  std::string name_;
};

class Constraint {
 public:
  explicit Constraint(
      const std::function<bool(const MatchContext&)>& constrain_fn)
      : IsContextMatchConstraint_(constrain_fn) {}
  bool operator()(const MatchContext& match_context) const {
    return IsContextMatchConstraint_(match_context);
  }

 private:
  std::function<bool(const MatchContext&)> IsContextMatchConstraint_;
};

class DrrPatternContext {
 public:
  DrrPatternContext();
  ~DrrPatternContext() = default;

  drr::SourcePattern SourcePattern();

  std::shared_ptr<SourcePatternGraph> source_pattern_graph() const {
    return source_pattern_graph_;
  }

  std::vector<Constraint> constraints() const;

  std::shared_ptr<ResultPatternGraph> result_pattern_graph() const {
    return result_pattern_graph_;
  }

 private:
  friend class drr::SourcePattern;
  friend class drr::ResultPattern;

  const Op& SourceOpPattern(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {});
  const drr::Tensor& SourceTensorPattern(const std::string& name);

  const Op& ResultOpPattern(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {});
  const drr::Tensor& ResultTensorPattern(const std::string& name);

  // void RequireEqual(const Attribute& first, const Attribute& second);
  void RequireEqual(const TensorShape& first, const TensorShape& second);

  std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  std::vector<Constraint> constraints_;
  std::shared_ptr<ResultPatternGraph> result_pattern_graph_;

  std::vector<std::shared_ptr<const drr::Op>> owned_ops_;
};

class Op {
 public:
  const std::string& name() const { return op_type_name_; }

  void operator()(const Tensor& arg, const Tensor* out) const;

  Tensor& operator()() const;

  Tensor& operator()(const Tensor& arg) const;
  Tensor& operator()(const Tensor& arg0, const Tensor& arg1) const;
  // const Tensor& operator()(const Tensor& arg0, const Tensor& arg1, const
  // Tensor& arg2) const; const Tensor& operator()(const Tensor& arg0, const
  // Tensor& arg1, const Tensor& arg2, const Tensor& arg3) const; const Tensor&
  // operator()(const Tensor& arg0, const Tensor& arg1, const Tensor& arg2,
  // const Tensor& arg3, const Tensor& arg4) const;
  // void operator()(const std::vector<Tensor>& args, const
  // std::vector<Tensor*>& outputs) const;

 private:
  friend class DrrPatternContext;

  Op(const std::string& op_type_name,
     const std::unordered_map<std::string, Attribute>& attributes,
     PatternGraph* pattern_graph)
      : op_type_name_(op_type_name),
        attributes_(attributes),
        pattern_graph_(pattern_graph) {}

  static int64_t count;

  std::string op_type_name_;
  std::unordered_map<std::string, Attribute> attributes_;
  PatternGraph* pattern_graph_;
};

class Tensor {
 public:
  const std::string& DebugName() const;

  TensorShape shape() const { return TensorShape(name()); }

  Tensor& operator=(const Tensor& other) = delete;

  void operator=(Tensor& other) const;  // NOLINT

  const std::string& name() const { return name_; }

  void set_name(const std::string& name) { name_ = name; }

  OpCall* producer() const { return producer_; }

  void set_producer(OpCall* producer) { producer_ = producer; }

  const std::vector<const OpCall*>& consumers() const { return consumers_; }

  void set_consumables(const std::vector<const OpCall*>& consumers) {
    consumers_ = consumers;
  }

  void AddConsumer(const OpCall* consumer) { consumers_.push_back(consumer); }

 private:
  friend class DrrPatternContext;
  friend class Op;

  Tensor(const std::string& name, PatternGraph* pattern_graph)
      : name_(name), pattern_graph_(pattern_graph) {}

  std::string name_;
  OpCall* producer_;
  std::vector<const OpCall*> consumers_;
  PatternGraph* pattern_graph_;
};

class OpCall {
 public:
  OpCall(const Op* op,
         const std::vector<const Tensor*>& inputs,
         const std::vector<const Tensor*>& outputs)
      : op_(op), inputs_(inputs), outputs_(outputs) {}

  const std::string& name() const { return op_->name(); }

  const std::vector<const Tensor*>& inputs() const { return inputs_; }

  const std::vector<const Tensor*>& outputs() const { return outputs_; }

 private:
  const Op* op_;
  std::vector<const Tensor*> inputs_;
  std::vector<const Tensor*> outputs_;
};

class ResultPattern {
 public:
  const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {}) {
    return ctx_->ResultOpPattern(op_type, attributes);
  }

  const drr::Tensor& Tensor(const std::string& name) {
    return ctx_->ResultTensorPattern(name);
  }

  Attribute Attr(const std::string& attr_name) { return Attribute(attr_name); }

 private:
  friend class SourcePattern;

  explicit ResultPattern(DrrPatternContext* ctx) : ctx_(ctx) {}

  DrrPatternContext* ctx_;
};

class SourcePattern {
 public:
  drr::ResultPattern ResultPattern() const { return drr::ResultPattern(ctx_); }

  const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {}) {
    return ctx_->SourceOpPattern(op_type, attributes);
  }

  const drr::Tensor& Tensor(const std::string& name) {
    return ctx_->SourceTensorPattern(name);
  }

  Attribute Attr(const std::string& attr_name) { return Attribute(attr_name); }

 private:
  friend class DrrPatternContext;
  explicit SourcePattern(DrrPatternContext* ctx) : ctx_(ctx) {}
  DrrPatternContext* ctx_;
};

}  // namespace drr
}  // namespace ir
