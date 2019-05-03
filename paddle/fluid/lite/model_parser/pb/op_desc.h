// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * This file implements a light-weight OpDesc like the framework::OpDesc. We
 * delete the unnecessary methods, and remove the underlying dependencies, such
 * as framework::Operator and boost::varient to make it runnable in mobile.
 */

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {
namespace pb {

using Attribute = variant<int, float, bool, std::vector<std::string>>;
using VariableNameMap = std::map<std::string, std::vector<std::string>>;

/*
 * The lite::OpDesc, an light-weight implementation of wrapper of proto::OpDesc.
 * Unlike the original one in framework::OpDesc, we remove the local members
 * except the desc_, to avoid the inconsistent state, which is normal in the
 * original interface and results in bugs.
 */
class OpDesc {
 public:
  OpDesc() {}

  OpDesc(const framework::proto::OpDesc &desc) : desc_(desc) {}

  void CopyFrom(const OpDesc &op_desc) { desc_ = op_desc.ReadonlyProto(); }

  framework::proto::OpDesc *Proto() { return &desc_; }
  const framework::proto::OpDesc &ReadonlyProto() const { return desc_; }

  std::string Type() const { return desc_.type(); }

  void SetType(const std::string &type) { desc_.set_type(type); }

  // Get the arguments of parameter called `param`
  std::vector<std::string> Input(const std::string &param) const {
    return GetArguments(desc_.inputs(), param);
  }

  std::vector<std::string> InputArgumentNames() const {
    return GetArgumentNames(desc_.inputs());
  }

  void SetInput(const std::string &param,
                const std::vector<std::string> &args) {
    SetArgument(desc_.mutable_inputs(), param, args);
  }

  std::vector<std::string> Output(const std::string &param) const {
    return GetArguments(desc_.outputs(), param);
  }

  std::vector<std::string> OutputArgumentNames() const {
    return GetArgumentNames(desc_.outputs());
  }

  void SetOutput(const std::string &param,
                 const std::vector<std::string> &args) {
    SetArgument(desc_.mutable_outputs(), param, args);
  }

  bool HasAttr(const std::string &name) const {
    const auto &xs = desc_.attrs();
    auto it = std::find_if(xs.begin(), xs.end(),
                           [&](const framework::proto::OpDesc_Attr &x) {
                             return x.name() == name;
                           });
    return it != xs.end();
  }

  framework::proto::AttrType GetAttrType(const std::string &name) const {
    const auto &xs = desc_.attrs();
    auto it = std::find_if(xs.begin(), xs.end(),
                           [&](const framework::proto::OpDesc_Attr &x) {
                             return x.name() == name;
                           });
    CHECK(it != xs.end());
    return it->type();
  }

  std::vector<std::string> AttrNames() const {
    std::vector<std::string> res;
    const auto &xs = desc_.attrs();
    std::transform(
        xs.begin(), xs.end(), std::back_inserter(res),
        [](const framework::proto::OpDesc_Attr &x) { return x.name(); });
    return res;
  }

  template <typename T>
  void SetAttr(const std::string &name, const T &v) {
    auto &xs = *desc_.mutable_attrs();
    auto it = std::find_if(xs.begin(), xs.end(),
                           [&](const framework::proto::OpDesc_Attr &x) {
                             return x.name() == name;
                           });
    if (it == xs.end()) {
      auto *attr = xs.Add();
      attr->set_name(name);
      it = std::find_if(xs.begin(), xs.end(),
                        [&](const framework::proto::OpDesc_Attr &x) {
                          return x.name() == name;
                        });
    }

    size_t hash = typeid(T).hash_code();
    if (hash == typeid(int).hash_code()) {
      it->set_type(framework::proto::INT);
      it->set_i(v);
    } else if (hash == typeid(float).hash_code()) {
      it->set_type(framework::proto::FLOAT);
      it->set_f(v);
    } else if (hash == typeid(bool).hash_code()) {
      it->set_type(framework::proto::BOOLEAN);
      it->set_b(v);
    } else {
      LOG(FATAL) << "unsupport attr type";
    }
  }

  Attribute GetAttr(const std::string &name) const {
    auto &xs = desc_.attrs();
    auto it = std::find_if(xs.begin(), xs.end(),
                           [&](const framework::proto::OpDesc_Attr &x) {
                             return x.name() == name;
                           });

    Attribute res;
    CHECK(it != xs.end());

    switch (it->type()) {
      case framework::proto::INT:
        res.set<int>(it->i());
        break;
      case framework::proto::FLOAT:
        res.set<float>(it->f());
        break;
      case framework::proto::STRING:
        res.set<std::string>(it->s());
        break;
      case framework::proto::BOOLEAN:
        res.set<bool>(it->b());
        break;

      default:
        LOG(FATAL) << "unsupported attr type";
    }

    return res;
  }

 private:
  std::vector<std::string> GetArguments(
      const google::protobuf::RepeatedPtrField<framework::proto::OpDesc_Var>
          &xs,
      const std::string &param) const {
    std::vector<std::string> res;
    auto it = std::find_if(xs.begin(), xs.end(),
                           [&](const framework::proto::OpDesc_Var &it) {
                             return it.parameter() == param;
                           });
    CHECK(it != xs.end());

    const auto &ys = it->arguments();
    std::transform(ys.begin(), ys.end(), std::back_inserter(res),
                   [](const std::string &x) { return x; });
    return res;
  }

  void SetArgument(
      google::protobuf::RepeatedPtrField<framework::proto::OpDesc_Var> *xs,
      const std::string &param, const std::vector<std::string> &args) {
    auto it = std::find_if(xs->begin(), xs->end(),
                           [&](const framework::proto::OpDesc_Var &it) {
                             return it.parameter() == param;
                           });
    if (it == xs->end()) {
      auto *new_arg = xs->Add();
      new_arg->set_parameter(param);
      for (const auto &arg : args) {
        *new_arg->mutable_arguments()->Add() = arg;
      }
    } else {
      it->mutable_arguments()->Clear();
      for (const auto &arg : args) {
        *it->mutable_arguments()->Add() = arg;
      }
    }
  }

  std::vector<std::string> GetArgumentNames(
      const google::protobuf::RepeatedPtrField<framework::proto::OpDesc_Var>
          &xs) const {
    std::vector<std::string> res;
    std::transform(
        xs.begin(), xs.end(), std::back_inserter(res),
        [](const framework::proto::OpDesc_Var &x) { return x.parameter(); });
    return res;
  }

 private:
  framework::proto::OpDesc desc_;
};

template <>
void OpDesc::SetAttr<std::string>(const std::string &name,
                                  const std::string &v);

}  // namespace pb
}  // namespace lite
}  // namespace paddle
