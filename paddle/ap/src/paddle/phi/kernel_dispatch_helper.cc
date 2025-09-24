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

#include "paddle/ap/include/paddle/phi/kernel_dispatch_helper.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/kernel_dispatch/builtin_frame_util.h"
#include "paddle/ap/include/kernel_dispatch/dispatch_ctx_method_class.h"
#include "paddle/ap/include/kernel_dispatch/value.h"
#include "paddle/ap/include/memory/guard.h"

namespace phi {

namespace {

using CoreExpr = ap::axpr::CoreExpr;
using Lambda = ap::axpr::Lambda<CoreExpr>;
using Val = ap::kernel_dispatch::Val;
using DispatchCtx = ap::kernel_dispatch::DispatchCtx<Val>;

}  // namespace

KernelDispatchHelper::KernelDispatchHelper()
    : circlable_ref_list_(ap::memory::Guard{}.circlable_ref_list()) {}

adt::Result<Val> KernelDispatchHelper::InterpretCtxMaker(
    const Lambda& ctx_maker_lambda) {
  ap::axpr::Interpreter cps_interpreter(
      ap::kernel_dispatch::MakeBuiltinFrameAttrMap<Val>(), circlable_ref_list_);
  ADT_LET_CONST_REF(ctx, cps_interpreter.Interpret(ctx_maker_lambda, {}));
  return ctx;
}

adt::Result<adt::Ok> KernelDispatchHelper::InterpretKernelDispatcher(
    const Lambda& kernel_dispatch_lambda, const DispatchCtx& dispatch_ctx) {
  const auto& cls = ap::kernel_dispatch::GetDispatchCtxClass<Val>();
  ap::axpr::BuiltinClassInstance<Val> instance{cls, dispatch_ctx};
  ap::axpr::Interpreter cps_interpreter(
      ap::kernel_dispatch::MakeBuiltinFrameAttrMap<Val>(), circlable_ref_list_);
  ADT_RETURN_IF_ERR(
      cps_interpreter.Interpret(kernel_dispatch_lambda, {instance}));
  return adt::Ok{};
}

}  // namespace phi
