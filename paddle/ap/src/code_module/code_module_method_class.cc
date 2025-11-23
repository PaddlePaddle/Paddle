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

#include "paddle/ap/include/code_module/code_module_method_class.h"

namespace ap::code_module {

template <typename ValueT>
struct TypeImplCodeModuleMethodClass {
  using This = TypeImplCodeModuleMethodClass;
  using Self = axpr::TypeImpl<CodeModule>;

  static adt::Result<CodeModule> Make(const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string("the constructor of 'CodeModule' takes 2 arguments. but ") +
        std::to_string(args.size()) + "were given."};
    const auto& list = args.at(0).Match(
        [&](const adt::List<ValueT>& l) -> adt::List<ValueT> { return l; },
        [&](const auto& impl) -> adt::List<ValueT> {
          return adt::List<ValueT>{ValueT{impl}};
        });
    adt::List<FuncDeclare> func_declares;
    func_declares->reserve(list->size());
    for (const auto& elt : *list) {
      ADT_LET_CONST_REF(func_declare, axpr::Get<FuncDeclare>(elt))
          << adt::errors::TypeError{
                 std::string() +
                 "the argument 1 of constructor of 'CodeModule' should be a "
                 "'FuncDeclare' object or a list of 'FuncDeclare' object."};
      func_declares->emplace_back(func_declare);
    }
    ADT_LET_CONST_REF(source_code, SourceCode::CastFromAxprValue(args.at(1)))
        << adt::errors::TypeError{std::string() +
                                  "the argument 2 of CodeModule() should be a "
                                  "'Project' (not " +
                                  axpr::GetTypeName(args.at(1)) + ") object"};
    return CodeModule{func_declares, source_code};
  }
};

template <typename ValueT>
adt::Result<ValueT> InitCodeModule(const ValueT& self_val,
                                   const std::vector<ValueT>& args) {
  ADT_LET_CONST_REF(
      empty_self,
      self_val.template TryGet<axpr::BuiltinClassInstance<ValueT>>());
  ADT_LET_CONST_REF(m, TypeImplCodeModuleMethodClass<ValueT>::Make(args));
  return empty_self.type.New(m);
}

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetCodeModuleClass() {
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "CodeModule", [&](const auto& DoEach) {
        DoEach("__init__", &InitCodeModule<axpr::Value>);
      }));
  using Self = CodeModule;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::code_module
