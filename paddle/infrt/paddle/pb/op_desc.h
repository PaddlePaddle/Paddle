// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <glog/logging.h>

#include "paddle/infrt/paddle/cpp/desc_api.h"
#include "paddle/infrt/paddle/framework.pb.h"
#include "paddle/infrt/support/variant.h"

namespace infrt {
namespace paddle {
namespace pb {

namespace framework_proto = ::paddle::framework::proto;

using Attribute =
    Variant<int, float, bool, std::vector<std::string>, std::vector<int>>;
using VariableNameMap = std::map<std::string, std::vector<std::string>>;

/*
 * The lite::OpDesc, an light-weight implementation of wrapper of proto::OpDesc.
 * Unlike the original one in framework::OpDesc, we remove the local members
 * except the desc_, to avoid the inconsistent state, which is normal in the
 * original interface and results in bugs.
 */
class OpDesc : public cpp::OpDescAPI {
 public:
  OpDesc() = delete;

  explicit OpDesc(framework_proto::OpDesc *desc) : desc_(desc) { CHECK(desc_); }

  framework_proto::OpDesc *Proto() { return desc_; }
  const framework_proto::OpDesc &ReadonlyProto() const { return *desc_; }

  std::string Type() const override { return desc_->type(); }

  void SetType(const std::string &type) override { desc_->set_type(type); }

  // Get the arguments of parameter called `param`
  std::vector<std::string> Input(const std::string &param) const override {
    return GetArguments(desc_->inputs(), param);
  }

  std::vector<std::string> InputArgumentNames() const override {
    return GetArgumentNames(desc_->inputs());
  }

  void SetInput(const std::string &param,
                const std::vector<std::string> &args) override {
    SetArgument(desc_->mutable_inputs(), param, args);
  }

  std::vector<std::string> Output(const std::string &param) const override {
    return GetArguments(desc_->outputs(), param);
  }

  std::vector<std::string> OutputArgumentNames() const override {
    return GetArgumentNames(desc_->outputs());
  }

  void SetOutput(const std::string &param,
                 const std::vector<std::string> &args) override {
    SetArgument(desc_->mutable_outputs(), param, args);
  }

  bool HasAttr(const std::string &name) const override {
    const auto &xs = desc_->attrs();
    auto it = std::find_if(
        xs.begin(), xs.end(), [&](const framework_proto::OpDesc_Attr &x) {
          return x.name() == name;
        });
    return it != xs.end();
  }

  AttrType GetAttrType(const std::string &name) const override {
    const auto &xs = desc_->attrs();
    auto it = std::find_if(
        xs.begin(), xs.end(), [&](const framework_proto::OpDesc_Attr &x) {
          return x.name() == name;
        });
    CHECK(it != xs.end());
#define DEF_ONE(type__)                   \
  case framework_proto::AttrType::type__: \
    return AttrType::type__;

    switch (it->type()) {
      DEF_ONE(INT);
      DEF_ONE(FLOAT);
      DEF_ONE(STRING);
      DEF_ONE(INTS);
      DEF_ONE(FLOATS);
      DEF_ONE(STRINGS);
      DEF_ONE(BOOLEAN);
      DEF_ONE(BOOLEANS);
      DEF_ONE(BLOCK);
      DEF_ONE(LONG);
      DEF_ONE(BLOCKS);
      DEF_ONE(LONGS);
      default:
        LOG(FATAL) << "Unknown attribute type";
        return static_cast<AttrType>(-1);
    }
#undef DEF_ONE
  }

  std::vector<std::string> AttrNames() const override {
    std::vector<std::string> res;
    const auto &xs = desc_->attrs();
    std::transform(
        xs.begin(),
        xs.end(),
        std::back_inserter(res),
        [](const framework_proto::OpDesc_Attr &x) { return x.name(); });
    return res;
  }

  template <typename T>
  void SetAttr(const std::string &name, const T &v);

  template <typename T>
  T GetAttr(const std::string &name) const;

 private:
  std::vector<std::string> GetArguments(
      const google::protobuf::RepeatedPtrField<framework_proto::OpDesc_Var> &xs,
      const std::string &param) const {
    std::vector<std::string> res;
    auto it = std::find_if(
        xs.begin(), xs.end(), [&](const framework_proto::OpDesc_Var &it) {
          return it.parameter() == param;
        });
    CHECK(it != xs.end());

    const auto &ys = it->arguments();
    std::transform(ys.begin(),
                   ys.end(),
                   std::back_inserter(res),
                   [](const std::string &x) { return x; });
    return res;
  }

  void SetArgument(
      google::protobuf::RepeatedPtrField<framework_proto::OpDesc_Var> *xs,
      const std::string &param,
      const std::vector<std::string> &args) {
    auto it = std::find_if(
        xs->begin(), xs->end(), [&](const framework_proto::OpDesc_Var &it) {
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
      const google::protobuf::RepeatedPtrField<framework_proto::OpDesc_Var> &xs)
      const {
    std::vector<std::string> res;
    std::transform(
        xs.begin(),
        xs.end(),
        std::back_inserter(res),
        [](const framework_proto::OpDesc_Var &x) { return x.parameter(); });
    return res;
  }

 private:
  framework_proto::OpDesc *desc_;
};

template <>
void OpDesc::SetAttr<std::string>(const std::string &name,
                                  const std::string &v);

template <>
void OpDesc::SetAttr<std::vector<int>>(const std::string &name,
                                       const std::vector<int> &v);

}  // namespace pb
}  // namespace paddle
}  // namespace infrt
