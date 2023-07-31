// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

namespace ir {
namespace drr {

class Op;
class Tensor;
class OpCall;
class Constrain;
class SourcePattern;
class ResultPattern;
class SourcePatternGraph;
class ResultPatternGraph;

using id_type = std::string;

class DrrPassContext : public std::enable_shared_from_this<DrrPassContext> {
 public:
  DrrPassContext() = default;
  ~DrrPassContext() = default;

  drr::SourcePattern SourcePattern() { return drr::SourcePattern(this); }

 private:
  friend class drr::SourcePattern;
  friend class drr::ResultPattern;

  const Op& SourceOpPattern(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {});
  const drr::Tensor& SourceTensorPattern(const std::string& tensor_id);

  const Op& ResultOpPattern(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {});
  const drr::Tensor& ResultTensorPattern(const std::string& tensor_id);

  std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  std::vector<std::unique_ptr<const Constrain>> constraints_;
  std::shared_ptr<ResultPatternGraph> result_pattern_graph_;

  std::vector<std::shared_ptr<const drr::Op>> owned_ops_;
};

class DrrPass {
 public:
  virtual void operator()(DrrPassContext* ctx) const;
};

class Attribute {
 public:
  explicit Attribute(const std::string& id) : attr_id_(id) {}

 private:
  std::string attr_id_;
};

class Op : public std::enable_shared_from_this<Op> {
 public:
  void operator()(const Tensor& arg, const Tensor* out) const {
    std::vector<std::weak_ptr<const Tensor>> inputs{arg.shared_from_this()};
    std::vector<std::weak_ptr<const Tensor>> outputs{out->shared_from_this()};
    add_op_call_fn_(
        std::make_shared<OpCall>(shared_from_this(), inputs, outputs));
  }

  // const Tensor& operator()(const Tensor& arg) const;
  // const Tensor& operator()(const Tensor& arg0, const Tensor& arg1) const;
  // const Tensor& operator()(const Tensor& arg0, const Tensor& arg1, const
  // Tensor& arg2) const; const Tensor& operator()(const Tensor& arg0, const
  // Tensor& arg1, const Tensor& arg2, const Tensor& arg3) const; const Tensor&
  // operator()(const Tensor& arg0, const Tensor& arg1, const Tensor& arg2,
  // const Tensor& arg3, const Tensor& arg4) const; void operator()(const
  // std::vector<const Tensor>& args, const std::vector<const Tensor*>& outputs)
  // const;

 private:
  friend class SourcePattern;

  Op(const std::string& op_type_name,
     const std::unordered_map<std::string, Attribute>& attributes,
     const std::function<void(std::shared_ptr<OpCall>)>& add_op_call_fn)
      : op_type_name_(op_type_name),
        attributes_(attributes),
        add_op_call_fn_(add_op_call_fn) {}

  std::string op_type_name_;
  std::unordered_map<std::string, Attribute> attributes_;
  std::function<void(std::shared_ptr<OpCall>)> add_op_call_fn_;
};

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  const std::string& DebugName() const;

  Tensor& operator=(const Tensor& other) = delete;

  void operator=(Tensor& other) const {  // NOLINT
    other.tensor_id_ = tensor_id_;
  }

  const id_type& id() const { return tensor_id_; }

  std::weak_ptr<OpCall> producer() const { return producer_; }

  void set_producer(std::weak_ptr<OpCall> producer) { producer_ = producer; }

  const std::unordered_set<std::weak_ptr<const OpCall>>& consumers() const {
    return consumers_;
  }

  void set_consumables(
      const std::unordered_set<std::weak_ptr<const OpCall>>& consumers) {
    consumers_ = consumers;
  }

  void AddConsumer(std::weak_ptr<const OpCall> consumer) {
    consumers_.insert(consumer);
  }

 private:
  friend class DrrPassContext;

  explicit Tensor(const id_type& tensor_id) : tensor_id_(tensor_id) {}

  id_type tensor_id_;
  std::weak_ptr<OpCall> producer_;
  std::unordered_set<std::weak_ptr<const OpCall>> consumers_;
};

class OpCall : public std::enable_shared_from_this<OpCall> {
 public:
  OpCall(std::weak_ptr<const Op> op,
         const std::vector<std::weak_ptr<const Tensor>>& inputs,
         const std::vector<std::weak_ptr<const Tensor>>& outputs)
      : op_(op), inputs_(inputs), outputs_(outputs) {}

  const std::vector<std::weak_ptr<const Tensor>>& inputs() const {
    return inputs_;
  }

  const std::vector<std::weak_ptr<const Tensor>>& outputs() const {
    return outputs_;
  }

 private:
  id_type op_call_id_;
  std::weak_ptr<const Op> op_;
  std::vector<std::weak_ptr<const Tensor>> inputs_;
  std::vector<std::weak_ptr<const Tensor>> outputs_;
};

class ResultPattern {
 public:
  const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {}) {
    return ctx_->ResultOpPattern(op_type, attributes);
  }

  const drr::Tensor& Tensor(const std::string& tensor_id) {
    return ctx_->ResultTensorPattern(tensor_id);
  }

  Attribute Attr(const std::string& attr_name) { return Attribute(attr_name); }

 private:
  friend class SourcePattern;

  explicit ResultPattern(DrrPassContext* ctx) : ctx_(ctx) {}

  DrrPassContext* ctx_;
};

class SourcePattern {
 public:
  ResultPattern ResultPattern() const { return ResultPattern(ctx_); }

  const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {}) {
    return ctx_->SourceOpPattern(op_type, attributes);
  }

  const drr::Tensor& Tensor(const std::string& tensor_id) {
    return ctx_->SourceTensorPattern(tensor_id);
  }

  Attribute Attr(const std::string& attr_name) { return Attribute(attr_name); }

 private:
  friend class DrrPassContext;
  explicit SourcePattern(DrrPassContext* ctx) : ctx_(ctx) {}
  DrrPassContext* ctx_;
};

}  // namespace drr
}  // namespace ir
