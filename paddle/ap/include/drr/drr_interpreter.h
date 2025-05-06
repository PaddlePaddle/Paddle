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
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/drr/value.h"
#include "paddle/ap/include/registry/abstract_drr_pass_registry_item.h"

namespace ap::drr {

class DrrInterpreter {
 public:
  explicit DrrInterpreter(
      const axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>&
          backend_ir_ctx,
      const std::weak_ptr<ap::memory::CirclableRefListBase>&
          circlable_ref_list);

  using Function = ap::axpr::Value;

  using DrrNode = ap::drr::Node;
  using DrrCtx = ap::drr::DrrCtx;

  ap::adt::Result<axpr::Value> Interpret(const Function& function,
                                         const std::vector<axpr::Value>& args) {
    return interpreter_.Interpret(function, args);
  }

  ap::adt::Result<DrrCtx> InterpretDrrCtxMaker(
      const Function& lambda, const std::vector<axpr::Value>& args);

  ap::adt::Result<DrrCtx> InterpretPass(
      const Function& function, const std::string& abstract_drr_pass_name);

  ap::adt::Result<DrrCtx> InterpretPass(
      const ap::axpr::ClassAttrs<ap::axpr::SerializableValue>& cls);

  ap::adt::Result<DrrCtx> CreateDrrCtxByDrrPassObj(
      const ap::axpr::Value& drr_pass_obj);

 private:
  axpr::Interpreter interpreter_;
};

}  // namespace ap::drr
