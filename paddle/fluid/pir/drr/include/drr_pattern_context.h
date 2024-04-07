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
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "paddle/fluid/pir/drr/include/drr_match_context.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
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

using ConstraintFunction = std::function<bool(const MatchContext&)>;
class Constraint {
 public:
  explicit Constraint(const ConstraintFunction& constrain_fn)
      : IsContextMatchConstraint_(constrain_fn) {}
  bool operator()(const MatchContext& match_context) const {
    return IsContextMatchConstraint_(match_context);
  }

 private:
  ConstraintFunction IsContextMatchConstraint_;
};

class TEST_API DrrPatternContext {
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
  drr::Tensor& SourceTensorPattern(const std::string& name);

  const Op& ResultOpPattern(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {});
  drr::Tensor& ResultTensorPattern(const std::string& name);

  // void RequireEqual(const Attribute& first, const Attribute& second);
  void RequireEqual(const TensorShape& first, const TensorShape& second);
  void RequireEqual(const TensorDataType& first, const TensorDataType& second);
  void RequireNativeCall(const ConstraintFunction& custom_fn);

  std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  std::vector<Constraint> constraints_;
  std::shared_ptr<ResultPatternGraph> result_pattern_graph_;

  std::vector<std::shared_ptr<const drr::Op>> owned_ops_;
};

class Op {
 public:
  TEST_API const std::string& name() const { return op_type_name_; }

  TEST_API Tensor& operator()() const;
  TEST_API void operator()(const Tensor& arg, const Tensor* out) const;
  TEST_API Tensor& operator()(const Tensor& arg) const;
  TEST_API Tensor& operator()(const Tensor& arg0, const Tensor& arg1) const;
  TEST_API Tensor& operator()(const Tensor& arg0,
                              const Tensor& arg1,
                              const Tensor& arg2) const;
  TEST_API void operator()(const std::vector<const Tensor*>& args,
                           const std::vector<const Tensor*>& outputs) const;
  // const Tensor& operator()(const Tensor& arg0, const Tensor& arg1, const
  // Tensor& arg2) const; const Tensor& operator()(const Tensor& arg0, const
  // Tensor& arg1, const Tensor& arg2, const Tensor& arg3) const; const Tensor&
  // operator()(const Tensor& arg0, const Tensor& arg1, const Tensor& arg2,
  // const Tensor& arg3, const Tensor& arg4) const;

  static const char* prefix;

 private:
  Op(const std::string& op_type_name,
     const std::unordered_map<std::string, Attribute>& attributes,
     PatternGraph* pattern_graph)
      : op_type_name_(op_type_name),
        attributes_(attributes),
        pattern_graph_(pattern_graph) {}

  const std::unordered_map<std::string, Attribute>& attributes() const {
    return attributes_;
  }

  friend class DrrPatternContext;
  friend class OpCall;

  std::string op_type_name_;
  std::unordered_map<std::string, Attribute> attributes_;
  PatternGraph* pattern_graph_{nullptr};

  thread_local static int64_t count;
};

class TEST_API Tensor {
 public:
  static const char RESULT_INPUT_NONE_TENSOR_NAME[];
  static const char RESULT_OUTPUT_NONE_TENSOR_NAME[];
  static const char SOURCE_INPUT_NONE_TENSOR_NAME[];
  static const char SOURCE_OUTPUT_NONE_TENSOR_NAME[];

  TensorShape shape() const { return TensorShape(name()); }

  TensorDataType dtype() const { return TensorDataType(name()); }

  bool is_none() const {
    return name_ == RESULT_INPUT_NONE_TENSOR_NAME ||
           name_ == RESULT_OUTPUT_NONE_TENSOR_NAME ||
           name_ == SOURCE_INPUT_NONE_TENSOR_NAME ||
           name_ == SOURCE_OUTPUT_NONE_TENSOR_NAME;
  }

  void Assign(const Tensor& other);

  void operator=(const Tensor& other) const;  // NOLINT

  const std::string& name() const { return name_; }

  void set_name(const std::string& name) { name_ = name; }

  OpCall* producer() const { return producer_; }

  void set_producer(OpCall* producer) { producer_ = producer; }

  const std::unordered_set<const OpCall*>& consumers() const {
    return consumers_;
  }

  void AddConsumer(const OpCall* consumer) { consumers_.insert(consumer); }

 private:
  Tensor(const std::string& name, PatternGraph* pattern_graph)
      : name_(name), pattern_graph_(pattern_graph) {}

  friend class DrrPatternContext;
  friend class Op;

  std::string name_;
  OpCall* producer_{nullptr};
  std::unordered_set<const OpCall*> consumers_;
  PatternGraph* pattern_graph_{nullptr};
};

class TEST_API OpCall {
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

class TEST_API ResultPattern {
 public:
  const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {});

  drr::Tensor& Tensor(const std::string& name);

  // Represent the input tensor which is none.
  // Example:
  // instance_norm has follow input tensor : (x, scale, bias), scale and
  // bias are optional(means it may be none).
  // When scale is none, we can write a instance_norm op in drr as follow:
  // res.Op("instance_norm")(res.Tensor("x"), res.InputNoneTensor(),
  // res.Tensor("bias"));
  drr::Tensor& InputNoneTensor();

  // Represent the output tensor which is none.
  // Example:
  // reshape has follow output tensor : (out, xshape), xshape is optional(means
  // it may be none). We can write a reshape op in drr as follow:
  // res.Op("reshape")({res.Tensor("x")}, {res.Tensor("out"),
  // res.OutputNoneTensor()});
  drr::Tensor& OutputNoneTensor();

  Attribute StrAttr(const std::string& value) const;

  Attribute BoolAttr(bool value) const;

  Attribute Int32Attr(int32_t value) const;

  Attribute Int64Attr(int64_t value) const;

  Attribute Float32Attr(float value) const;

  Attribute VectorInt64Attr(const std::vector<int64_t>& value) const;

  Attribute VectorInt32Attr(const std::vector<int32_t>& value) const;

  Attribute VectorFloatAttr(const std::vector<float>& value) const;

  Attribute DataTypeAttr(const std::string& value) const;

  Attribute PlaceAttr(const std::string& value) const;

  Attribute DataLayoutAttr(const std::string& value) const;

  Attribute ComputeAttr(const AttrComputeFunc& attr_compute_func) const;

 private:
  friend class SourcePattern;

  explicit ResultPattern(DrrPatternContext* ctx) : ctx_(ctx) {}

  DrrPatternContext* ctx_{nullptr};
};

class TEST_API SourcePattern {
 public:
  drr::ResultPattern ResultPattern() const;

  const drr::Op& Op(
      const std::string& op_type,
      const std::unordered_map<std::string, Attribute>& attributes = {});

  const drr::Tensor& Tensor(const std::string& name);

  Attribute Attr(const std::string& attr_name) const;

  void RequireEqual(const TensorShape& first, const TensorShape& second);

  void RequireEqual(const TensorDataType& first, const TensorDataType& second);

  void RequireNativeCall(const ConstraintFunction& custom_fn);

  // Same as a ResultPattern::InputNoneTensor
  drr::Tensor& InputNoneTensor();

  // Same as a ResultPattern::OutputNoneTensor
  drr::Tensor& OutputNoneTensor();

 private:
  friend class DrrPatternContext;
  explicit SourcePattern(DrrPatternContext* ctx) : ctx_(ctx) {}
  DrrPatternContext* ctx_{nullptr};
};

}  // namespace drr
}  // namespace paddle
