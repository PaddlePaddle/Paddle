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
#include <functional>
#include <map>
#include <string>
#include <unordered_map>

#include "paddle/framework/attribute.h"
#include "paddle/framework/ddim.h"

namespace paddle {
namespace framework {
class OperatorBase;
using VariableNameMap = std::map<std::string, std::vector<std::string>>;

using OpCreator = std::function<OperatorBase*(
    const std::string& /*type*/, const VariableNameMap& /*inputs*/,
    const VariableNameMap& /*outputs*/, const AttributeMap& /*attrs*/)>;

class InferShapeContextBase {
 public:
  virtual ~InferShapeContextBase() {}
  virtual const DDim get_input_dim(const std::string& name) const = 0;
  virtual void set_input_dim(const std::string& name,
                             const DDim& dim) const = 0;
  virtual const DDim get_output_dim(const std::string& name) const = 0;
  virtual void set_output_dim(const std::string& name,
                              const DDim& dim) const = 0;

 protected:
  virtual const DDim get_dim(const std::string& name) const = 0;
  virtual void set_dim(const std::string& name, const DDim& dim) const = 0;
};

// this class not only make proto but also init attribute checkers.
class OpProtoAndCheckerMaker {
 public:
  OpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : proto_(proto), op_checker_(op_checker) {}

  virtual ~OpProtoAndCheckerMaker() {
    PADDLE_ENFORCE(validated_, "should call Validate after build");
  }

  void Validate();

  virtual void InferShape(const InferShapeContextBase& ctx) const = 0;

 protected:
  struct VariableBuilder {
    OpProto::Var* var_;

    VariableBuilder& AsDuplicable() {
      var_->set_duplicable(true);
      return *this;
    }

    VariableBuilder& AsIntermediate() {
      var_->set_intermediate(true);
      return *this;
    }

    VariableBuilder& NotInGradient() {
      var_->set_not_in_gradient(true);
      return *this;
    }
  };

  VariableBuilder AddInput(const std::string& name, const std::string& comment);

  VariableBuilder AddOutput(const std::string& name,
                            const std::string& comment);

  template <typename T>
  TypedAttrChecker<T>& AddAttr(const std::string& name,
                               const std::string& comment,
                               bool generated = false) {
    auto* attr = proto_->add_attrs();
    attr->set_name(name);
    attr->set_comment(comment);
    attr->set_generated(generated);
    attr->set_type(AttrTypeID<T>());
    return op_checker_->AddAttrChecker<T>(name);
  }

  void AddComment(const std::string& comment) { proto_->set_comment(comment); }

 private:
  void CheckNoDuplicatedInOutAttrs();

  OpProto* proto_;
  OpAttrChecker* op_checker_;
  bool validated_{false};
};

class NOPMaker : public OpProtoAndCheckerMaker {
 public:
  NOPMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {}
};

struct OpInfo {
  OpCreator creator_;
  std::string grad_op_type_;
  OpProto* proto_;
  OpAttrChecker* checker_;
  OpProtoAndCheckerMaker* maker_;

  bool HasOpProtoAndChecker() const {
    return proto_ != nullptr && checker_ != nullptr;
  }

  const OpProto& Proto() const {
    PADDLE_ENFORCE_NOT_NULL(proto_, "Operator Proto has not been registered");
    PADDLE_ENFORCE(proto_->IsInitialized(),
                   "Operator Proto must be initialized in op info");
    return *proto_;
  }

  const OpAttrChecker& Checker() const {
    PADDLE_ENFORCE_NOT_NULL(checker_,
                            "Operator Checker has not been registered");
    return *checker_;
  }

  const OpCreator& Creator() const {
    PADDLE_ENFORCE_NOT_NULL(creator_,
                            "Operator Creator has not been registered");
    return creator_;
  }

  const OpProtoAndCheckerMaker& Maker() const { return *maker_; }

  bool HasGradientOp() const { return !grad_op_type_.empty(); }
};

class OpInfoMap {
 public:
  static OpInfoMap& Instance();

  OpInfoMap(const OpInfoMap& o) = delete;
  OpInfoMap(OpInfoMap&& o) = delete;
  OpInfoMap& operator=(const OpInfoMap& o) = delete;
  OpInfoMap& operator=(OpInfoMap&& o) = delete;

  bool Has(const std::string& op_type) const {
    return map_.find(op_type) != map_.end();
  }

  void Insert(const std::string& type, const OpInfo& info) {
    PADDLE_ENFORCE(!Has(type), "Operator %s has been registered", type);
    map_.insert({type, info});
  }

  const OpInfo& Get(const std::string& type) const {
    auto op_info_ptr = GetNullable(type);
    PADDLE_ENFORCE_NOT_NULL(op_info_ptr, "Operator %s has not been registered",
                            type);
    return *op_info_ptr;
  }

  const OpInfo* GetNullable(const std::string& type) const {
    auto it = map_.find(type);
    if (it == map_.end()) {
      return nullptr;
    } else {
      return &it->second;
    }
  }

  template <typename Callback>
  void IterAllInfo(Callback callback) {
    for (auto& it : map_) {
      callback(it.first, it.second);
    }
  }

 private:
  OpInfoMap() = default;
  std::unordered_map<std::string, const OpInfo> map_;
};

}  // namespace framework
}  // namespace paddle
