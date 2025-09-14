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
#include "paddle/ap/include/code_gen/code_gen_ctx.h"
#include "paddle/ap/include/code_gen/code_gen_result.h"
#include "paddle/ap/include/code_gen/value.h"
#include "paddle/ap/include/code_module/code_module.h"
#include "paddle/ap/include/paddle/pir_node.h"

namespace ap::paddle {

struct ApKernelDefineHelper {
  std::weak_ptr<ap::memory::CirclableRefListBase> circlable_ref_list_;

  explicit ApKernelDefineHelper(
      const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list)
      : circlable_ref_list_(circlable_ref_list) {}

  using Function = ap::axpr::Value;
  using CodeModule = ap::code_module::CodeModule;
  using PirNode = ap::paddle::PirNode;
  using CGValue = ap::code_gen::Value;
  using CodeGenCtx = ap::code_gen::CodeGenCtx<PirNode>;
  using CodeGenResult = ap::code_gen::CodeGenResult<CGValue>;

  ap::adt::Result<CodeGenResult> Interpret(const Function& lambda,
                                           const CodeGenCtx& code_gen_ctx);
};

}  // namespace ap::paddle
