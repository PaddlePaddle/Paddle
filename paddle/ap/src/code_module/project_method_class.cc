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

#include "paddle/ap/include/code_module/project_method_class.h"

namespace ap::code_module {

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetProjectClass();

struct TypeProjectClassMethodClass {
  static adt::Result<axpr::Value> New(
      const axpr::Value&, const std::vector<axpr::Value>& args_vec) {
    const auto& packed_args = axpr::CastToPackedArgs(args_vec);
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->empty()) << adt::errors::TypeError{
        std::string() + "Project() takes no positional argument, bug " +
        std::to_string(args->size()) + " were given"};
    axpr::AttrMap<File> dentry2file{};
    ADT_LET_CONST_REF(directory_val, kwargs->Get("nested_files"))
        << adt::errors::TypeError{
               std::string() +
               "Project() need the keyword argument 'nested_files'"};
    ADT_LET_CONST_REF(directory,
                      directory_val.template CastTo<Directory<File>>())
        << adt::errors::TypeError{
               std::string() +
               "the keyword argument 'nested_files' of Project() should be a "
               "Project.Directory, but " +
               axpr::GetTypeName(directory_val) + " were given"};
    ADT_LET_CONST_REF(compile_cmd_val, kwargs->Get("compile_cmd"))
        << adt::errors::TypeError{
               std::string() +
               "Project() need the keyword argument 'compile_cmd'"};
    ADT_LET_CONST_REF(compile_cmd,
                      compile_cmd_val.template CastTo<std::string>())
        << adt::errors::TypeError{std::string() +
                                  "the keyword argument 'compile_cmd' of "
                                  "Project() should be a str, but " +
                                  axpr::GetTypeName(compile_cmd_val) +
                                  " were given"};
    ADT_LET_CONST_REF(so_relative_path_val, kwargs->Get("so_relative_path"))
        << adt::errors::TypeError{
               std::string() +
               "Project() need the keyword argument 'so_relative_path'"};
    ADT_LET_CONST_REF(so_relative_path,
                      so_relative_path_val.template CastTo<std::string>())
        << adt::errors::TypeError{std::string() +
                                  "the keyword argument 'so_relative_path' of "
                                  "Project() should be a str, but " +
                                  axpr::GetTypeName(so_relative_path_val) +
                                  " were given"};
    axpr::AttrMap<axpr::SerializableValue> others;
    if (kwargs->Has("others")) {
      ADT_LET_CONST_REF(others_val, kwargs->Get("others"))
          << adt::errors::TypeError{
                 std::string() +
                 "Project() need the keyword argument 'others'"};
      ADT_LET_CONST_REF(
          others_attrs,
          others_val.template CastTo<axpr::AttrMap<axpr::SerializableValue>>())
          << adt::errors::TypeError{
                 std::string() +
                 "the keyword argument 'others' of Project() should be a "
                 "BuiltinSerializableAttrMap, but " +
                 axpr::GetTypeName(others_val) + " were given"};
      others = others_attrs;
    }
    return GetProjectClass().New(
        Project{directory, compile_cmd, so_relative_path, others});
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetProjectClass() {
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("Project", [&](const auto& DoEach) {
        DoEach("__init__", &TypeProjectClassMethodClass::New);
        DoEach("FileContent", GetFileContentClass());
        DoEach("SoftLink", GetSoftLinkClass());
        DoEach("Directory", GetDirectoryClass());
      }));
  return axpr::MakeGlobalNaiveClassOps<Project>(cls);
}

}  // namespace ap::code_module
