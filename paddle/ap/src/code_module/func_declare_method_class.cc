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

#include "paddle/ap/include/code_module/func_declare_method_class.h"

namespace ap::code_module {

template <typename ValueT>
struct TypeImplFuncDeclareMethodClass {
  using This = TypeImplFuncDeclareMethodClass;
  using Self = axpr::TypeImpl<FuncDeclare>;

  static adt::Result<FuncDeclare> Make(const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 3) << adt::errors::TypeError{
        std::string("the constructor of FuncDeclare takes 3 arguments but ") +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(ret_type, CastToArgType<axpr::Value>(args.at(0)))
        << adt::errors::TypeError{std::string() +
                                  "the argument 1 of FuncDeclare() should be a "
                                  "'DataType or PointerType'"};
    ADT_LET_CONST_REF(func_id, axpr::TryGetImpl<std::string>(args.at(1)))
        << adt::errors::TypeError{std::string() +
                                  "the argument 2 of "
                                  "FuncDeclare() should be a 'str'"};
    ADT_LET_CONST_REF(arg_types, GetArgTypes(args.at(2)));
    return FuncDeclare{ret_type, func_id, arg_types};
  }

  static Result<adt::List<ArgType>> GetArgTypes(const ValueT& val) {
    ADT_LET_CONST_REF(list, axpr::TryGetImpl<adt::List<ValueT>>(val))
        << adt::errors::TypeError{std::string() +
                                  "the argument 2 of construct of FuncDeclare "
                                  "should be a list of DataType "
                                  "or PointerType."};
    adt::List<ArgType> ret;
    ret->reserve(list->size());
    for (const auto& elt : *list) {
      ADT_LET_CONST_REF(arg_type, CastToArgType(elt))
          << adt::errors::TypeError{std::string() +
                                    "the argument 2 of construct of "
                                    "FuncDeclare should be a list of DataType "
                                    "or PointerType."};
      ret->emplace_back(arg_type);
    }
    return ret;
  }
};

template <typename ValueT>
adt::Result<ValueT> InitFuncDeclare(const ValueT& self_val,
                                    const std::vector<ValueT>& args) {
  ADT_LET_CONST_REF(
      empty_self,
      self_val.template TryGet<axpr::BuiltinClassInstance<ValueT>>());
  ADT_LET_CONST_REF(func_declare,
                    TypeImplFuncDeclareMethodClass<ValueT>::Make(args));
  return empty_self.type.New(func_declare);
}

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetFuncDeclareClass() {
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "FuncDeclare", [&](const auto& DoEach) {
        DoEach("__init__", &InitFuncDeclare<axpr::Value>);
      }));
  using Self = FuncDeclare;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::code_module
