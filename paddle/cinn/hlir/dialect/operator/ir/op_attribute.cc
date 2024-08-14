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

#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"

namespace cinn {
namespace dialect {

const cinn::fusion::FusionTrackerPtr &FusionTrackerPtrAttribute::data() const {
  return storage()->GetAsKey();
}

const GroupInfo &GroupInfoAttribute::data() const {
  return storage()->GetAsKey();
}

const cinn::hlir::framework::pir::CINNKernelInfo &
CINNKernelInfoAttribute::data() const {
  return storage()->GetAsKey();
}
}  // namespace dialect
}  // namespace cinn

IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::GroupInfoAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::CINNKernelInfoAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::FusionTrackerPtrAttribute)
