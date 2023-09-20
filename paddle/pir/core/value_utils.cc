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

#include "paddle/pir/core/value_utils.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/pir/core/op_result.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"

namespace pir {
std::string GetValueInfo(Value v) {
  std::stringstream ss;
  ss << "op name=" << v.dyn_cast<OpResult>().owner()->name();
  ss << ", index=" << v.dyn_cast<OpResult>().index();
  ss << ", dtype=" << v.type();
  if (v.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
    ss << ", place="
       << v.type()
              .dyn_cast<paddle::dialect::AllocatedDenseTensorType>()
              .place();
  }
  return ss.str();
}

}  // namespace pir
