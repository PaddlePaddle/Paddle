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
#include "paddle/ap/include/axpr/frame.h"
#include "paddle/ap/include/axpr/serializable_value.h"

namespace ap::axpr {

template <typename ValueT>
class Environment {
 public:
  virtual adt::Result<ValueT> Get(const std::string& var) const = 0;

  virtual adt::Result<adt::Ok> Set(const std::string& var,
                                   const ValueT& val) = 0;

  virtual std::optional<Frame<SerializableValue>> GetConstGlobalFrame() const {
    return std::nullopt;
  }

  virtual std::optional<Frame<SerializableValue>>
  RecursivelyGetConstGlobalFrame() const = 0;

 protected:
  Environment() = default;
  Environment(const Environment&) = delete;
  Environment(Environment&&) = delete;
};

}  // namespace ap::axpr
