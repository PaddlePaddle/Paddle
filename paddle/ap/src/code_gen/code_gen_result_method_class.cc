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

#include "paddle/ap/include/code_gen/code_gen_result_method_class.h"

namespace ap::code_gen {

template <typename ValueT>
struct TypeImplCodeGenResultMethodClass {
  using This = TypeImplCodeGenResultMethodClass;
  using Self = axpr::TypeImpl<CodeGenResult<ValueT>>;

  static adt::Result<ValueT> Construct(const ValueT& self_val,
                                       const std::vector<ValueT>& args) {
    return This{}.Make(self_val, args);
  }

  adt::Result<ValueT> Make(const ValueT& self_val,
                           const std::vector<ValueT>& packed_args_val) {
    ADT_LET_CONST_REF(
        empty_self,
        self_val.template TryGet<axpr::BuiltinClassInstance<ValueT>>());
    const auto& packed_args = axpr::CastToPackedArgs(packed_args_val);
    const auto& [args, kwargs] = *packed_args;
    ADT_LET_CONST_REF(module_val, kwargs->Get("module"))
        << adt::errors::TypeError{
               std::string() +
               "the constructor of 'CodeGenResult' missing keyword argument "
               "'module' of type 'CodeModule'."};
    ADT_LET_CONST_REF(m, axpr::Get<code_module::CodeModule>(module_val))
        << adt::errors::TypeError{
               std::string() +
               "the constructor of 'CodeGenResult' missing keyword argument "
               "'module' of type 'CodeModule'."};
    ADT_LET_CONST_REF(
        kernel_dispatch_func,
        kwargs->template TryGet<axpr::Function<axpr::SerializableValue>>(
            "kernel_dispatch_func"))
        << adt::errors::TypeError{
               std::string() +
               "the constructor of 'CodeGenResult' missing keyword argument "
               "'kernel_dispatch_func' of type 'Function'."};
    std::optional<axpr::AttrMap<axpr::SerializableValue>>
        kernel_dispatch_const_data;
    if (kwargs->Has("kernel_dispatch_const_data")) {
      ADT_LET_CONST_REF(
          data,
          kwargs->template TryGet<axpr::AttrMap<axpr::SerializableValue>>(
              "kernel_dispatch_const_data"))
          << adt::errors::TypeError{
                 std::string() +
                 "the constructor of 'CodeGenResult' needs keyword argument "
                 "'kernel_dispatch_const_data' of type "
                 "'BuiltinSerializableAttrMap'."};
      kernel_dispatch_const_data = data;
    } else {
      kernel_dispatch_const_data = axpr::AttrMap<axpr::SerializableValue>{};
    }
    ADT_CHECK(kernel_dispatch_const_data.has_value());
    return empty_self.type.New(CodeGenResult<ValueT>{
        m, kernel_dispatch_func, kernel_dispatch_const_data.value()});
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetCodeGenResultClass() {
  using TypeImplMethods = TypeImplCodeGenResultMethodClass<axpr::Value>;
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "CodeGenResult", [&](const auto& Define) {
        Define("__init__", &TypeImplMethods::Construct);
      }));
  using Self = CodeGenResult<axpr::Value>;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::code_gen
