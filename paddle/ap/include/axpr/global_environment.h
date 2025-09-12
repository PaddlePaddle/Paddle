// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/environment.h"

namespace ap::axpr {

template <typename ValueT>
class GlobalEnvironment : public Environment<ValueT> {
 public:
  adt::Result<ValueT> Get(const std::string& var) const override {
    ADT_LET_CONST_REF(frame_ptr, frame_.Get());
    const auto& res = frame_ptr->OptGet(var);
    if (res.has_value()) {
      return res.value();
    }
    if (parent_ == nullptr) {
      return NameError{std::string("name '") + var + "' is not defined."};
    }
    return parent_->Get(var);
  }

  adt::Result<adt::Ok> Set(const std::string& var, const ValueT& val) override {
    ADT_LET_CONST_REF(frame_ptr, frame_.Mut());
    {
      static std::string tmp_var_prefix("__");
      if (var.substr(0, tmp_var_prefix.size()) != tmp_var_prefix) {
        ADT_CHECK(SerializableValue::IsSerializable(val)) << [&] {
          std::ostringstream ss;
          ss << "Only serializable values are supported insert into global "
                "environment. " ss
             << "Builtin serializable types are: ";
          ss << SerializableValue::SerializableTypeNames();
          ss << " (not include '" << axpr::GetTypeName(val) << "').";
          return adt::errors::ValueError{ss.str()};
        }();
      }
    }
    frame_ptr->Set(var, val);
    return adt::Ok{};
  }

  const Frame<ValueT>& frame() const { return frame_; }

  GlobalEnvironment(const std::shared_ptr<Environment<ValueT>>& parent,
                    const Frame<ValueT>& frame)
      : parent_(parent), frame_(frame) {}

 private:
  GlobalEnvironment(const GlobalEnvironment&) = delete;
  GlobalEnvironment(GlobalEnvironment&&) = delete;

  std::shared_ptr<Environment<ValueT>> parent_;
  Frame<ValueT> frame_;
};

}  // namespace ap::axpr
