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
#include "paddle/ap/include/axpr/frame.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/axpr/serializable_value_helper.h"

namespace ap::axpr {

template <typename ValueT>
class MutableGlobalEnvironment : public Environment<ValueT> {
 public:
  MutableGlobalEnvironment(const std::shared_ptr<Environment<ValueT>>& parent,
                           const Frame<SerializableValue>& const_frame,
                           const Frame<ValueT>& temp_frame)
      : parent_(parent), const_frame_(const_frame), temp_frame_(temp_frame) {}

  adt::Result<ValueT> Get(const std::string& var) const override {
    if (IsTempVar(var)) {
      ADT_LET_CONST_REF(temp_frame_ptr, temp_frame_.Get());
      const auto& val_in_temp_frame = temp_frame_ptr->OptGet(var);
      if (val_in_temp_frame.has_value()) {
        return val_in_temp_frame.value();
      }
    } else {
      ADT_LET_CONST_REF(const_frame_ptr, const_frame_.Get());
      const auto& val_in_const_frame = const_frame_ptr->OptGet(var);
      if (val_in_const_frame.has_value()) {
        return val_in_const_frame.value().template CastTo<ValueT>();
      }
    }
    if (parent_ == nullptr) {
      return NameError{std::string("name '") + var + "' is not defined."};
    }
    return parent_->Get(var);
  }

  adt::Result<adt::Ok> Set(const std::string& var, const ValueT& val) override {
    if (IsTempVar(var)) {
      ADT_LET_CONST_REF(temp_frame_ptr, temp_frame_.Mut());
      temp_frame_ptr->Set(var, val);
    } else {
      ADT_LET_CONST_REF(const_frame_ptr, const_frame_.Mut());
      SerializableValueHelper helper{};
      ADT_LET_CONST_REF(serializable_val, helper.CastFrom(val)) << [&] {
        std::ostringstream ss;
        ss << "Only serializable values are supported insert into global "
              "environment. ";
        ss << "Builtin serializable types are: ";
        ss << SerializableValue::SerializableTypeNames();
        ss << " (not include '" << axpr::GetTypeName(val) << "').";
        return adt::errors::ValueError{ss.str()};
      }();
      const_frame_ptr->Set(var, serializable_val);
    }
    return adt::Ok{};
  }

  bool IsTempVar(const std::string& var) const {
    static std::string tmp_var_prefix("__");
    return var.substr(0, tmp_var_prefix.size()) == tmp_var_prefix;
  }

  std::optional<Frame<SerializableValue>> GetConstGlobalFrame() const override {
    return const_frame_;
  }

  std::optional<Frame<SerializableValue>> RecursivelyGetConstGlobalFrame()
      const override {
    return const_frame_;
  }

  const Frame<ValueT>& temp_frame() const { return temp_frame_; }

 private:
  MutableGlobalEnvironment(const MutableGlobalEnvironment&) = delete;
  MutableGlobalEnvironment(MutableGlobalEnvironment&&) = delete;

  std::shared_ptr<Environment<ValueT>> parent_;
  Frame<SerializableValue> const_frame_;
  Frame<ValueT> temp_frame_;
};

}  // namespace ap::axpr
