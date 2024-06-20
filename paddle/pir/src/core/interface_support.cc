// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/include/core/interface_support.h"

namespace pir {
InterfaceValue::InterfaceValue(InterfaceValue&& val) noexcept {
  type_id_ = val.type_id_;
  model_ = std::move(val.model_);
}

InterfaceValue& InterfaceValue::operator=(InterfaceValue&& val) noexcept {
  swap(std::move(val));
  return *this;
}
}  // namespace pir
