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

#include "paddle/ap/include/code_module/soft_link_method_class.h"

namespace ap::code_module {

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetSoftLinkClass();

struct TypeSoftLinkMethodClass {
  static adt::Result<axpr::Value> New(const axpr::Value&,
                                      const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "SoftLink() takes 1 argument, but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(target_relative_path,
                      args.at(0).template CastTo<std::string>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of SoftLink() should a a str, but " +
               axpr::GetTypeName(args.at(0)) + " were given"};
    return GetSoftLinkClass().New(SoftLink{target_relative_path});
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetSoftLinkClass() {
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("SoftLink", [&](const auto& DoEach) {
        DoEach("__init__", &TypeSoftLinkMethodClass::New);
      }));
  return axpr::MakeGlobalNaiveClassOps<SoftLink>(cls);
}

}  // namespace ap::code_module
