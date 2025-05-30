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
#include "paddle/ap/include/code_module/api_wrapper_project_maker.h"
#include "paddle/ap/include/code_module/code_module.h"
#include "paddle/ap/include/code_module/package.h"
#include "paddle/ap/include/code_module/project_compile_helper.h"

namespace ap::code_module {

class ModuleCompileHelper {
  std::string workspace_dir_;
  std::string relative_dir_in_workspace_;

 public:
  ModuleCompileHelper(const std::string& workspace_dir,
                      const std::string& relative_dir_in_workspace)
      : workspace_dir_(workspace_dir),
        relative_dir_in_workspace_(relative_dir_in_workspace) {}

  adt::Result<CodeModule> CompileProjectModuleToPackageModule(
      const CodeModule& project_module) const {
    ADT_CHECK(project_module->source_code.template Has<Project>());
    const auto& func_declares = project_module->func_declares.vector();
    ADT_LET_CONST_REF(
        api_wrapper_project,
        code_module::ApiWrapperProjectMaker{}.Make(func_declares));
    ADT_LET_CONST_REF(main_project, GetMainProject(project_module));
    code_module::ProjectCompileHelper api_wrapper_compile_helper(
        GetApiWrapperProjectAbsoluteDir(), api_wrapper_project);
    code_module::ProjectCompileHelper main_compile_helper(
        GetMainProjectAbsoluteDir(), main_project);
    ADT_RETURN_IF_ERR(api_wrapper_compile_helper.DumpNestedFilesToFs());
    ADT_RETURN_IF_ERR(main_compile_helper.DumpNestedFilesToFs());
    ADT_RETURN_IF_ERR(api_wrapper_compile_helper.Compile());
    ADT_RETURN_IF_ERR(main_compile_helper.Compile());
    std::string api_wrapper_so_relative_path =
        GetApiWrapperProjectRelativeDir() + "/" +
        api_wrapper_project->so_relative_path;
    std::string main_so_relative_path =
        GetMainProjectRelativeDir() + "/" + main_project->so_relative_path;
    Package ret_package{
        /*nested_files=*/Directory<File>{},
        /*api_wrapper_so_relative_path=*/api_wrapper_so_relative_path,
        /*main_so_relative_path=*/main_so_relative_path,
        /*others=*/axpr::AttrMap<axpr::SerializableValue>{}};
    return CodeModule{project_module->func_declares, ret_package};
  }

 private:
  adt::Result<code_module::Project> GetMainProject(
      const code_module::CodeModule& code_module) const {
    return code_module->source_code.template TryGet<code_module::Project>();
  }

  std::string GetApiWrapperProjectAbsoluteDir() const {
    return workspace_dir_ + "/" + GetApiWrapperProjectRelativeDir();
  }

  std::string GetMainProjectAbsoluteDir() const {
    return workspace_dir_ + "/" + GetMainProjectRelativeDir();
  }

  std::string GetApiWrapperProjectRelativeDir() const {
    return relative_dir_in_workspace_ + "/api_wrapper/";
  }
  std::string GetMainProjectRelativeDir() const {
    return relative_dir_in_workspace_ + "/main/";
  }
};

}  // namespace ap::code_module
