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
#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/environment.h"

namespace ap::axpr {

template <typename ValueT>
class BuiltinEnvironment : public Environment<ValueT> {
 public:
  explicit BuiltinEnvironment(const AttrMap<ValueT>& builtin_object)
      : builtin_object_(builtin_object) {}

  adt::Result<ValueT> Get(const std::string& var) const override {
    return builtin_object_->Get(var);
  }

  adt::Result<adt::Ok> Set(const std::string& var, const ValueT& val) override {
    return adt::errors::RuntimeError{"builtin environment is immutable."};
  }

  std::optional<Frame<SerializableValue>> RecursivelyGetConstGlobalFrame()
      const override {
    return std::nullopt;
  }

 private:
  BuiltinEnvironment(const BuiltinEnvironment&) = delete;
  BuiltinEnvironment(BuiltinEnvironment&&) = delete;

  AttrMap<ValueT> builtin_object_;
};

}  // namespace ap::axpr
