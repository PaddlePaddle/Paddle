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

#include "paddle/framework/attribute.h"
#include "paddle/framework/framework.pb.h"

namespace paddle {
namespace framework {

// this class not only make proto but also init attribute checkers.
class OpProtoAndCheckerMaker {
 public:
  OpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : proto_(proto), op_checker_(op_checker) {}

  virtual ~OpProtoAndCheckerMaker() {
    PADDLE_ENFORCE(validated_, "should call Validate after build");
  }

  void Validate();

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

    VariableBuilder& AsDispensable() {
      var_->set_dispensable(true);
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

}  // namespace framework
}  // namespace paddle
