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

#include "paddle/ap/include/code_module/directory_method_class.h"

namespace ap::code_module {

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetDirectoryClass();

struct TypeDirectoryClassMethodClass {
  static adt::Result<axpr::Value> New(const axpr::Value&,
                                      const std::vector<axpr::Value>& args) {
    axpr::AttrMap<File> dentry2file{};
    int i = 0;
    for (const auto& arg : args) {
      ++i;
      ADT_LET_CONST_REF(pair, arg.template CastTo<adt::List<axpr::Value>>())
          << adt::errors::TypeError{
                 std::string() + "the argument of " + std::to_string(i) +
                 " Directory() should be a [str, Project.Directory | "
                 "Project.FileContent | Project.SoftLink]"
                 ", but " +
                 axpr::GetTypeName(arg) + " were given"};
      ADT_CHECK(pair->size() == 2) << adt::errors::TypeError{
          std::string() + "the argument of " + std::to_string(i) +
          " Directory() should be a [str, Project.Directory | "
          "Project.FileContent | Project.SoftLink]"
          ", but its length is " +
          std::to_string(pair->size())};
      ADT_LET_CONST_REF(dentry, pair->at(0).template CastTo<std::string>())
          << adt::errors::TypeError{
                 std::string() + "the argument of " + std::to_string(i) +
                 " Directory() only accepts list of [str, Project.Directory | "
                 "Project.FileContent | Project.SoftLink]"
                 ". but the first of pair is a " +
                 axpr::GetTypeName(pair->at(0))};
      ADT_LET_CONST_REF(file, File::CastFromAxprValue(pair->at(1)))
          << adt::errors::TypeError{
                 std::string() +
                 "Directory() only accepts list of [str, Project.Directory | "
                 "Project.FileContent | Project.SoftLink]"
                 ", but the second of pair is a " +
                 axpr::GetTypeName(pair->at(1))};
      dentry2file->Set(dentry, file);
    }
    return GetDirectoryClass().New(Directory<File>{dentry2file});
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetDirectoryClass() {
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("Directory", [&](const auto& DoEach) {
        DoEach("__init__", &TypeDirectoryClassMethodClass::New);
      }));
  return axpr::MakeGlobalNaiveClassOps<Directory<File>>(cls);
}

}  // namespace ap::code_module
