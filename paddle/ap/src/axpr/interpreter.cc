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
#include "paddle/ap/include/fs/builtin_functions.h"

namespace ap::axpr {

// xtrans/clang's `if constexpr` branch inside a member-template lambda
// (see CpsInterpreter::InterpretBuiltinSymbolCall in cps_interpreter.h,
// which dispatches to `this->template InterpretBuiltinUnarySymbolCall<Sym>`
// / `InterpretBuiltinBinarySymbolCall<Sym>` per builtin_symbol type) does
// not implicitly instantiate these member function templates the way
// GCC/system-clang do; the call sites end up as unresolved external
// references at link time. Force explicit instantiation for every
// builtin_symbol type used by CpsInterpreter so the definitions are
// actually emitted into this translation unit.
#define AP_EXPLICIT_INSTANTIATE_UNARY(Sym)      \
  template adt::Result<adt::Ok>                 \
  CpsInterpreter::InterpretBuiltinUnarySymbolCall<builtin_symbol::Sym>( \
      ComposedCallImpl<axpr::Value>*);
#define AP_EXPLICIT_INSTANTIATE_BINARY(Sym)      \
  template adt::Result<adt::Ok>                  \
  CpsInterpreter::InterpretBuiltinBinarySymbolCall<builtin_symbol::Sym>( \
      ComposedCallImpl<axpr::Value>*);

AP_EXPLICIT_INSTANTIATE_UNARY(Not)
AP_EXPLICIT_INSTANTIATE_UNARY(Neg)
AP_EXPLICIT_INSTANTIATE_UNARY(Starred)
AP_EXPLICIT_INSTANTIATE_UNARY(Call)
AP_EXPLICIT_INSTANTIATE_UNARY(ToString)
AP_EXPLICIT_INSTANTIATE_UNARY(Hash)
AP_EXPLICIT_INSTANTIATE_UNARY(Length)

AP_EXPLICIT_INSTANTIATE_BINARY(Add)
AP_EXPLICIT_INSTANTIATE_BINARY(Sub)
AP_EXPLICIT_INSTANTIATE_BINARY(Mul)
AP_EXPLICIT_INSTANTIATE_BINARY(Div)
AP_EXPLICIT_INSTANTIATE_BINARY(FloorDiv)
AP_EXPLICIT_INSTANTIATE_BINARY(Mod)
AP_EXPLICIT_INSTANTIATE_BINARY(EQ)
AP_EXPLICIT_INSTANTIATE_BINARY(NE)
AP_EXPLICIT_INSTANTIATE_BINARY(GT)
AP_EXPLICIT_INSTANTIATE_BINARY(GE)
AP_EXPLICIT_INSTANTIATE_BINARY(LT)
AP_EXPLICIT_INSTANTIATE_BINARY(LE)
AP_EXPLICIT_INSTANTIATE_BINARY(GetAttr)
AP_EXPLICIT_INSTANTIATE_BINARY(SetAttr)
AP_EXPLICIT_INSTANTIATE_BINARY(GetItem)
AP_EXPLICIT_INSTANTIATE_BINARY(SetItem)

#undef AP_EXPLICIT_INSTANTIATE_UNARY
#undef AP_EXPLICIT_INSTANTIATE_BINARY

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
