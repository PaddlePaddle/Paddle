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
#include "paddle/ap/include/code_module/project_compile_helper.h"
#include "paddle/ap/include/rt_module/naive_dl_handler.h"
#include "paddle/ap/include/rt_module/naive_module.h"

namespace ap::rt_module {

struct NaiveModuleMaker {
  explicit NaiveModuleMaker(const std::string& workspace_dir_val)
      : workspace_dir(workspace_dir_val) {}

  adt::Result<std::shared_ptr<const Module>> Make(
      const code_module::CodeModule& code_module,
      const std::function<const std::string&(const code_module::CodeModule&)>&
          Serialize) const {
    using RetT = adt::Result<std::shared_ptr<const Module>>;
    return code_module->source_code.Match(
        [&](const code_module::Project&) -> RetT {
          return MakeByProject(code_module, Serialize);
        },
        [&](const code_module::Package&) -> RetT {
          return MakeByPackage(code_module, Serialize);
        });
  }

  adt::Result<std::shared_ptr<const Module>> MakeByProject(
      const code_module::CodeModule& code_module,
      const std::function<const std::string&(const code_module::CodeModule&)>&
          Serialize) const {
    const auto& func_declares = code_module->func_declares.vector();
    ADT_LET_CONST_REF(
        api_wrapper_project,
        code_module::ApiWrapperProjectMaker{}.Make(func_declares));
    ADT_LET_CONST_REF(main_project, GetMainProject(code_module));
    code_module::ProjectCompileHelper api_wrapper_compile_helper(
        GetApiWrapperProjectDir(), api_wrapper_project);
    code_module::ProjectCompileHelper main_compile_helper(GetMainProjectDir(),
                                                          main_project);
    const auto& serialized_project = Serialize(code_module);
    if (FileExists(GetSerializedProjectFilePath())) {
      ADT_LET_CONST_REF(dumped, ReadSerializedProject());
      ADT_CHECK(dumped == serialized_project);
    } else {
      ADT_RETURN_IF_ERR(api_wrapper_compile_helper.DumpNestedFilesToFs());
      ADT_RETURN_IF_ERR(main_compile_helper.DumpNestedFilesToFs());
      ADT_RETURN_IF_ERR(api_wrapper_compile_helper.Compile());
      ADT_RETURN_IF_ERR(main_compile_helper.Compile());
      ADT_RETURN_IF_ERR(WriteSerializedProject(serialized_project));
    }
    std::string api_wrapper_so_path = api_wrapper_compile_helper.GetSoPath();
    std::string main_so_path = main_compile_helper.GetSoPath();
    ADT_LET_CONST_REF(dl_handler,
                      NaiveDlHandle::DlOpen(main_so_path, api_wrapper_so_path));
    return NaiveModule::Make(func_declares, dl_handler);
  }

  adt::Result<std::shared_ptr<const Module>> MakeByPackage(
      const code_module::CodeModule& code_module,
      const std::function<const std::string&(const code_module::CodeModule&)>&
          Serialize) const {
    const auto& func_declares = code_module->func_declares.vector();
    ADT_LET_CONST_REF(package, GetPackage(code_module));
    std::string api_wrapper_so_path =
        GetPackageDir() + "/" + package->api_wrapper_so_relative_path;
    ADT_CHECK(FileExists(api_wrapper_so_path)) << adt::errors::TypeError{
        std::string() +
        "FileExists(api_wrapper_so_path) failed. api_wrapper_so_path: " +
        api_wrapper_so_path};
    std::string main_so_path =
        GetPackageDir() + "/" + package->main_so_relative_path;
    ADT_CHECK(FileExists(main_so_path)) << adt::errors::TypeError{
        std::string() +
        "FileExists(main_so_path) failed. main_so_path: " + main_so_path};
    ADT_LET_CONST_REF(dl_handler,
                      NaiveDlHandle::DlOpen(main_so_path, api_wrapper_so_path));
    return NaiveModule::Make(func_declares, dl_handler);
  }

  adt::Result<std::string> ReadSerializedProject() const {
    std::ifstream ifs(GetSerializedProjectFilePath());
    ADT_CHECK(ifs.is_open());
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return buffer.str();
  }

  bool FileExists(const std::string& filepath) const {
    std::fstream fp;
    fp.open(filepath, std::fstream::in);
    if (fp.is_open()) {
      fp.close();
      return true;
    } else {
      return false;
    }
  }

  adt::Result<adt::Ok> WriteSerializedProject(
      const std::string& serialized_project) const {
    std::ofstream ofs(GetSerializedProjectFilePath());
    ADT_CHECK(ofs.is_open());
    ofs << serialized_project;
    ofs.close();
    return adt::Ok{};
  }

  adt::Result<code_module::Project> GetMainProject(
      const code_module::CodeModule& code_module) const {
    return code_module->source_code.template TryGet<code_module::Project>();
  }

  adt::Result<code_module::Package> GetPackage(
      const code_module::CodeModule& code_module) const {
    return code_module->source_code.template TryGet<code_module::Package>();
  }

  std::string GetApiWrapperProjectDir() const {
    return workspace_dir + "/api_wrapper/";
  }

  std::string GetPackageDir() const { return workspace_dir; }

  std::string GetMainProjectDir() const { return workspace_dir + "/main/"; }

  std::string GetSerializedProjectFilePath() const {
    return workspace_dir + "/serialized_project.json";
  }

 private:
  std::string workspace_dir;
};

}  // namespace ap::rt_module
