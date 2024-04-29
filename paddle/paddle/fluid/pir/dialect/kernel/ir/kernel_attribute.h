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

#pragma once

#include "paddle/fluid/pir/dialect/kernel/ir/attribute_storage.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/attribute.h"

namespace paddle {
namespace dialect {

class KernelAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(KernelAttribute, KernelAttributeStorage);

  bool operator<(const KernelAttribute &right) const {
    return storage() < right.storage();
  }

  static std::string name() { return "a_kernel"; }

  phi::KernelKey data() const { return storage()->GetAsKey(); }
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::KernelAttribute)
