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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/class_attrs.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/code_gen/ir_op.h"
#include "paddle/ap/include/code_gen/op_code_gen_ctx.h"

namespace ap::code_gen {

template <typename BirNode>
struct OpCudaCodeGenImpl {
  adt::Result<std::string> CodeGen(const OpCodeGenCtx<BirNode>& op_code_gen_ctx,
                                   const IrOp<BirNode>& ir_op);
  adt::Result<axpr::ClassAttrs<axpr::SerializableValue>>
  ConvertFusionOpToClassAttrs(const OpCodeGenCtx<BirNode>& op_code_gen_ctx,
                              const IrOp<BirNode>& ir_op);
};

}  // namespace ap::code_gen
