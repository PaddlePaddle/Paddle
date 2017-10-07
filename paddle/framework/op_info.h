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
#include "paddle/framework/op_desc.h"
#include "paddle/framework/type_defs.h"
#include "paddle/platform/macros.h"

namespace paddle {
namespace framework {

struct OpInfo {
  OpCreator creator_;
  GradOpMakerFN grad_op_maker_;
  OpProto* proto_{nullptr};
  OpAttrChecker* checker_{nullptr};

  bool HasOpProtoAndChecker() const {
    return proto_ != nullptr && checker_ != nullptr;
  }

  const OpProto& Proto() const {
    PADDLE_ENFORCE_NOT_NULL(proto_, "Operator Proto has not been registered");
    PADDLE_ENFORCE(proto_->IsInitialized(),
                   "Operator Proto must be initialized in op info");
    return *proto_;
  }

  const OpCreator& Creator() const {
    PADDLE_ENFORCE_NOT_NULL(creator_,
                            "Operator Creator has not been registered");
    return creator_;
  }

  const GradOpMakerFN& GradOpMaker() const {
    PADDLE_ENFORCE_NOT_NULL(grad_op_maker_,
                            "Operator GradOpMaker has not been registered.");
    return grad_op_maker_;
  }

  const OpAttrChecker* Checker() const { return checker_; }
};

class OpInfoMap {
 public:
  static OpInfoMap& Instance();

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

  DISABLE_COPY_AND_ASSIGN(OpInfoMap);
};

}  // namespace framework
}  // namespace paddle
