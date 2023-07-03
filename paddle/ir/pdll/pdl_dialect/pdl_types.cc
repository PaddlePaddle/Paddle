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

#include "paddle/ir/pdll/pdl_dialect/pdl_types.h"
#include "paddle/ir/pdll/pdl_dialect/pdl_dialect.h"

namespace ir {
namespace pdl {}
}  // namespace ir

IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDLType)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::AttributeType)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::OperationType)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::TypeType)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::ValueType)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::RangeType)
