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

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "paddle/fluid/pir/drr/api/match_context.h"

namespace pir {
namespace drr {

class Op;
class Tensor;
class OpCall;
class SourcePattern;
class ResultPattern;
class PatternGraph;
class SourcePatternGraph;
class ResultPatternGraph;

class NormalAttribute {
 public:
  explicit NormalAttribute(const std::string& name) : attr_name_(name) {}

  const std::string& name() const { return attr_name_; }

 private:
  std::string attr_name_;
};

using AttrComputeFunc = std::function<std::any(const MatchContext&)>;

class ComputeAttribute {
 public:
  explicit ComputeAttribute(const AttrComputeFunc& attr_compute_func)
      : attr_compute_func_(attr_compute_func) {}

  const AttrComputeFunc& attr_compute_func() const {
    return attr_compute_func_;
  }

 private:
  AttrComputeFunc attr_compute_func_;
};

using Attribute = std::variant<NormalAttribute, ComputeAttribute>;

class TensorShape {
 public:
  explicit TensorShape(const std::string& tensor_name)
      : tensor_name_(tensor_name) {}

  const std::string& tensor_name() const { return tensor_name_; }

 private:
  std::string tensor_name_;
};

class TensorDataType {
 public:
  explicit TensorDataType(const std::string& tensor_name)
      : tensor_name_(tensor_name) {}

  const std::string& tensor_name() const { return tensor_name_; }

 private:
  std::string tensor_name_;
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
  drr::Tensor& ResultTensorPattern(const std::string& name);

  // void RequireEqual(const Attribute& first, const Attribute& second);
  void RequireEqual(const TensorShape& first, const TensorShape& second);
  void RequireEqual(const TensorDataType& first, const TensorDataType& second);
  void RequireNativeCall(
      const std::function<bool(const MatchContext&)>& custom_fn);

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
  void operator()(const std::vector<const Tensor*>& args,
                  const std::vector<const Tensor*>& outputs) const;
  // const Tensor& operator()(const Tensor& arg0, const Tensor& arg1, const
  // Tensor& arg2) const; const Tensor& operator()(const Tensor& arg0, const
  // Tensor& arg1, const Tensor& arg2, const Tensor& arg3) const; const Tensor&
  // operator()(const Tensor& arg0, const Tensor& arg1, const Tensor& arg2,
  // const Tensor& arg3, const Tensor& arg4) const;

  static const char* prefix;

 private:
  friend class DrrPatternContext;
  friend class OpCall;

  Op(const std::string& op_type_name,
     const std::unordered_map<std::string, Attribute>& attributes,
     PatternGraph* pattern_graph)
      : op_type_name_(op_type_name),
        attributes_(attributes),
        pattern_graph_(pattern_graph) {}

  const std::unordered_map<std::string, Attribute>& attributes() const {
    return attributes_;
  }

  thread_local static int64_t count;

  std::string op_type_name_;
  std::unordered_map<std::string, Attribute> attributes_;
  PatternGraph* pattern_graph_{nullptr};
};

class Tensor {
 public:
  static const char NONE_TENSOR_NAME[];

  const std::string& DebugName() const;

  TensorShape shape() const { return TensorShape(name()); }

  TensorDataType dtype() const { return TensorDataType(name()); }

  bool is_none() const { return name_ == NONE_TENSOR_NAME; }

  void Assign(const Tensor& other);

  void operator=(const Tensor& other) const;  // NOLINT

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
  OpCall* producer_{nullptr};
  std::vector<const OpCall*> consumers_;
  PatternGraph* pattern_graph_{nullptr};
};

class OpCall {
 public:
  OpCall(const Op* op,
         const std::vector<const Tensor*>& inputs,
         const std::vector<const Tensor*>& outputs)
      : op_name_(op->op_type_name_),
        inputs_(inputs),
        outputs_(outputs),
        attributes_(op->attributes_) {}

  const std::string& name() const { return op_name_; }

  const std::vector<const Tensor*>& inputs() const { return inputs_; }

  const std::vector<const Tensor*>& outputs() const { return outputs_; }

  const std::unordered_map<std::string, Attribute>& attributes() const {
    return attributes_;
  }

 private:
  std::string op_name_;
  std::vector<const Tensor*> inputs_;
  std::vector<const Tensor*> outputs_;
  std::unordered_map<std::string, Attribute> attributes_;
};

class ResultPattern {
 public:
  const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {}) {
    return ctx_->ResultOpPattern(op_type, attributes);
  }

  drr::Tensor& Tensor(const std::string& name) {
    return ctx_->ResultTensorPattern(name);
  }

  // Represent the input tensor which is none.
  // Example:
  // instance_norm has follow input tensor : (x, scale, bias), scale and
  // bias are optional(means it may be none).
  // When scale is onoe, we can write a instance_norm op in drr as follow:
  // res.Op("instance_norm")(res.Tensor("x"), res.NoneTensor,
  // res.Tensor("bias"));
  drr::Tensor& NoneTensor() {
    return ctx_->ResultTensorPattern(Tensor::NONE_TENSOR_NAME);
  }

  Attribute Attr(const std::string& attr_name) const {
    return NormalAttribute(attr_name);
  }
  Attribute Attr(const AttrComputeFunc& attr_compute_func) const {
    return ComputeAttribute(attr_compute_func);
  }

 private:
  friend class SourcePattern;

  explicit ResultPattern(DrrPatternContext* ctx) : ctx_(ctx) {}

  DrrPatternContext* ctx_{nullptr};
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

  Attribute Attr(const std::string& attr_name) const {
    return NormalAttribute(attr_name);
  }

  void RequireEqual(const TensorShape& first, const TensorShape& second) {
    ctx_->RequireEqual(first, second);
  }
  void RequireEqual(const TensorDataType& first, const TensorDataType& second) {
    ctx_->RequireEqual(first, second);
  }

  void RequireNativeCall(
      const std::function<bool(const MatchContext&)>& custom_fn) {
    ctx_->RequireNativeCall(custom_fn);
  }

 private:
  friend class DrrPatternContext;
  explicit SourcePattern(DrrPatternContext* ctx) : ctx_(ctx) {}
  DrrPatternContext* ctx_{nullptr};
};

}  // namespace drr
}  // namespace pir
