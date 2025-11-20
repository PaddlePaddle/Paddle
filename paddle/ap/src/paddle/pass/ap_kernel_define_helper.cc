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

#include "paddle/ap/include/paddle/pass/ap_kernel_define_helper.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/code_gen/builtin_frame_util.h"
#include "paddle/ap/include/code_gen/value.h"
#include "paddle/ap/include/code_gen/value_method_class.h"
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/paddle/pir_graph_descriptor.h"
#include "paddle/ap/include/paddle/pir_node_descriptor.h"
#include "paddle/ap/include/paddle/pir_node_method_class.h"

namespace ap::paddle {

namespace adt = ap::adt;

namespace {

using Function = ap::axpr::Value;
using CodeModule = ap::code_module::CodeModule;
using PirNode = ap::paddle::PirNode;
using Val = ap::code_gen::Value;
using CodeGenCtx = ap::code_gen::CodeGenCtx<PirNode>;
using CodeGenResult = ap::code_gen::CodeGenResult<Val>;

}  // namespace

adt::Result<CodeGenResult> ApKernelDefineHelper::Interpret(
    const Function& lambda, const CodeGenCtx& code_gen_ctx) {
  ap::axpr::BuiltinClassInstance<CGValue> code_gen_ctx_instance{
      ap::code_gen::GetCodeGenCtxClass<CGValue, PirNode>(), code_gen_ctx};
  ap::axpr::Interpreter interpreter(
      ap::code_gen::MakeBuiltinFrameAttrMap<Val>(), circlable_ref_list_);
  ADT_CHECK(code_gen_ctx->ir_match_ctx.has_value());
  const auto& ir_match_ctx = code_gen_ctx->ir_match_ctx.value();
  ap::ir_match::OpMatchCtx<PirNode> op_match_ctx{ir_match_ctx.shared_ptr()};
  ap::axpr::BuiltinClassInstance<CGValue> op_match_ctx_instance{
      ap::ir_match::GetOpMatchCtxClass<CGValue, PirNode>(), op_match_ctx};
  ap::ir_match::TensorMatchCtx<PirNode> tensor_match_ctx{
      ir_match_ctx.shared_ptr()};
  ap::axpr::BuiltinClassInstance<CGValue> tensor_match_ctx_instance{
      ap::ir_match::GetTensorMatchCtxClass<CGValue, PirNode>(),
      tensor_match_ctx};
  ADT_LET_CONST_REF(result,
                    interpreter.Interpret(lambda,
                                          {code_gen_ctx_instance,
                                           op_match_ctx_instance,
                                           tensor_match_ctx_instance}));
  ADT_LET_CONST_REF(m, result.template CastTo<CodeGenResult>());
  return m;
}

}  // namespace ap::paddle
