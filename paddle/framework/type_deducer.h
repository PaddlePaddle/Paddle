/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include <map>
#include <memory>
#include <typeindex>
#include <vector>

#include "paddle/framework/attr_checker.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/tensor.h"
#include "paddle/framework/variable.h"

namespace paddle {
namespace framework {

/*
 * Operator will get different types of tensors, most operators just treat some
 * complex tensor types as basic tensor, but the meta-info of complex tensor may
 * need to be stored.
 *
 * For example, a LOTTensor stores a basic tensor and a LOT info, when it is
 * transfered to a FC operator as a input, the FC operator just visit the basic
 * tensor, but the LOT info should be copied to its outputs, in case of some
 * operator consuming its outputs need the LOT info. FC knows nothing about LOT,
 * so the framework should copy the LOT info transparently.
 *
 * Whether the operator's outputs should be a complex tensor, for example, a
 * FC's outputs are basic tensors, but they will be LOTTensor or SparseTensor if
 * FC's inputs are one of these types.
 *
 * In a word, we need a API(called TypeDeducer here) to automatically deduce
 * tensor types for operators, get its context (inputs' types, operator's
 * attributes) and the target tensor type it wants.
 */
class TypeDeduceRule;
class TypeDeducer {
 public:
  void Init(std::vector<const Variable*>* op_inputs, AttributeMap* op_attrs) {
    this->op_inputs = op_inputs;
    this->op_attrs = op_attrs;
  }

  void AddRule(const std::type_info& type,
               std::unique_ptr<TypeDeduceRule>&& rule) {
    auto it = rules.find(type);
    if (it == rules.end()) {
      rules[type] = std::vector<std::unique_ptr<TypeDeduceRule>>();
    }
    rules[type].emplace_back(std::move(rule));
  }

  static TypeDeducer& Global() {
    static auto x = std::unique_ptr<TypeDeducer>(new TypeDeducer());
    return *x;
  }

  /*
   * variable's name is passed so that we can get its additional attributes
   * defined in op_attrs to help deduce its tensor type.
   */
  template <typename Target>
  Target* operator()(const std::string& var_name) const;

  // store all the context.
  std::vector<const Variable*>* op_inputs;
  AttributeMap* op_attrs;
  Scope* scope;
  std::map<std::type_index /*target type*/,
           std::vector<std::unique_ptr<TypeDeduceRule>>>
      rules;

 private:
  explicit TypeDeducer();
};

/*
 * Base class of type deduce rules.
 */
class TypeDeduceRule {
 public:
  void Init(const TypeDeducer* deducer) {
    this->op_inputs = deducer->op_inputs;
    this->op_attrs = deducer->op_attrs;
    this->scope = deducer->scope;
  }

  /*
   * @param[in] var_name: variable's name
   * @param[in] target_type: the tensor type caller wants
   */
  virtual Variable* operator()(const std::string& var_name,
                               const std::type_info& target_type) const = 0;
  /*
   * Whether this rule matches.
   */
  virtual bool Match() const = 0;

  virtual ~TypeDeduceRule() {}

 protected:
  std::vector<const Variable*>* op_inputs;
  AttributeMap* op_attrs;
  Scope* scope;
};

/*
 * Default deduce rule, if all the inputs are Tensor, then the output's tensors
 * are Tensor.
 */
class Tensor2TensorDeduceRule final : public TypeDeduceRule {
 public:
  virtual Variable* operator()(
      const std::string& var_name,
      const std::type_info& target_type) const override {
    PADDLE_ENFORCE(target_type.hash_code() == typeid(Tensor).hash_code(),
                   "wrong rule on target type %s", target_type.name());
    auto target = scope->CreateVariable(var_name);
    target->GetMutable<Tensor>();
    return target;
  }

  virtual bool Match() const override {
    for (const auto var : *op_inputs) {
      if (!var->IsType<Tensor>()) {
        return false;
      }
    }
    return true;
  }
};

/*
 * Default deduce rule for LOTTensor(used in RNNOp), if one of the inputs is
 * LODTensor, all the outputs will be LODTensor.
 */
class LOT2TensorDeduceRule final : public TypeDeduceRule {
 public:
  virtual Variable* operator()(
      const std::string& var_name,
      const std::type_info& target_type) const override {
    PADDLE_ENFORCE(target_type.hash_code() == typeid(Tensor).hash_code(),
                   "wrong rule on target type %s", target_type.name());
    auto target = scope->CreateVariable(var_name);
    auto lodtensor = target->GetMutable<LODTensor>();
    lodtensor->ShareConstLODFrom(*first_lot_tensor_);
    return target;
  }

  virtual bool Match() const override {
    for (auto var : *op_inputs) {
      if (var->IsType<LODTensor>()) {
        first_lot_tensor_ = const_cast<LODTensor*>(&var->Get<LODTensor>());
        return true;
      }
    }

    first_lot_tensor_ = nullptr;
    return false;
  }

 private:
  mutable LODTensor* first_lot_tensor_;
};

}  // namespace framework
}  // namespace paddle
