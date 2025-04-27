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

#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/cps_interpreter.h"

namespace ap::axpr {

adt::Result<axpr::Value> Interpreter::Interpret(
    const Lambda<CoreExpr>& lambda, const std::vector<axpr::Value>& args) {
  CpsInterpreter cps_interpreter{builtin_frame_attr_map_, circlable_ref_list_};
  return cps_interpreter.Interpret(lambda, args);
}

adt::Result<axpr::Value> Interpreter::Interpret(
    const axpr::Value& function, const std::vector<axpr::Value>& args) {
  CpsInterpreter cps_interpreter{builtin_frame_attr_map_, circlable_ref_list_};
  return cps_interpreter.Interpret(function, args);
}

adt::Result<axpr::Value> Interpreter::InterpretModule(
    const Frame<SerializableValue>& const_global_frame,
    const Lambda<CoreExpr>& lambda) {
  CpsInterpreter cps_interpreter{builtin_frame_attr_map_, circlable_ref_list_};
  return cps_interpreter.InterpretModule(const_global_frame, lambda);
}

}  // namespace ap::axpr
