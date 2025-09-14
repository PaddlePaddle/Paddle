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

#include "paddle/ap/include/paddle/pass/ap_drr_helper.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/drr/builtin_frame_util.h"
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/drr_interpreter.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/drr/value_method_class.h"
#include "paddle/ap/include/paddle/pir/pir_method_class.h"

namespace ap::paddle {

namespace adt = ap::adt;

namespace {

using Function = ap::axpr::Value;

using DrrNode = ap::drr::Node;
using DrrCtx = ap::drr::DrrCtx;

}  // namespace

ApDrrHelper::ApDrrHelper(
    const std::weak_ptr<ap::memory::CirclableRefListBase>& circlable_ref_list)
    : drr_interpreter_(circlable_ref_list) {}

adt::Result<DrrCtx> ApDrrHelper::InterpretDrrCtxMaker(
    const Function& lambda, const std::vector<ap::axpr::Value>& args) {
  return drr_interpreter_.InterpretDrrCtxMaker(lambda, args);
}

adt::Result<DrrCtx> ApDrrHelper::Interpret(const Function& lambda,
                                           const std::string& drr_pass_name) {
  return drr_interpreter_.InterpretPass(lambda, drr_pass_name);
}

adt::Result<DrrCtx> ApDrrHelper::CreateDrrCtxByDrrPassObj(
    const ap::axpr::Value& obj) {
  return drr_interpreter_.CreateDrrCtxByDrrPassObj(obj);
}

adt::Result<DrrCtx> ApDrrHelper::Interpret(
    const ap::axpr::ClassAttrs<ap::axpr::SerializableValue>& cls) {
  return drr_interpreter_.InterpretPass(cls);
}

}  // namespace ap::paddle
