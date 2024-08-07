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

#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"

namespace pir {
bool ContainerType::classof(Type type) { return StackType::classof(type); }

}  // namespace pir
IR_DEFINE_EXPLICIT_TYPE_ID(pir::ContainerType)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::StackType)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::InletType)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::OutletType)
