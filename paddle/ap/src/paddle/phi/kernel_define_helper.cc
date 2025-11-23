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

#include "paddle/ap/include/paddle/phi/kernel_define_helper.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/code_module/builtin_frame_util.h"
#include "paddle/ap/include/code_module/value.h"
#include "paddle/ap/include/code_module/value_method_class.h"
#include "paddle/ap/include/memory/guard.h"

namespace phi {

namespace {

using CoreExpr = ap::axpr::CoreExpr;

using Lambda = ap::axpr::Lambda<CoreExpr>;

using CodeModule = ap::code_module::CodeModule;

using Val = ap::code_module::Value;

}  // namespace

adt::Result<CodeModule> KernelDefineHelper::InterpretKernelDefineLambda(
    const Lambda& lambda) {
  ap::memory::Guard guard{};
  ap::axpr::Interpreter cps_interpreter(
      ap::code_module::MakeBuiltinFrameAttrMap<Val>(),
      guard.circlable_ref_list());
  ADT_LET_CONST_REF(interpret_ret, cps_interpreter.Interpret(lambda, {}));
  ADT_LET_CONST_REF(m, ap::axpr::Get<CodeModule>(interpret_ret));
  return m;
}

}  // namespace phi
