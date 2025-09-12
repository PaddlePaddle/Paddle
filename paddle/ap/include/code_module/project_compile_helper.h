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

#include <sys/wait.h>
#include <fstream>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/code_module/project.h"
#include "paddle/ap/include/env/ap_path.h"

namespace ap::code_module {

struct ProjectCompileHelper {
  ProjectCompileHelper(const std::string& workspace_dir_val,
                       const Project& project_val)
      : workspace_dir(workspace_dir_val), project(project_val) {}

  adt::Result<adt::Ok> DumpNestedFilesToFs() {
    return DumpNestedFilesToFs(this->project->nested_files, "");
  }

  adt::Result<adt::Ok> Compile() {
    int ret_code = 0;
    std::string change_dir_cmd = std::string() + "cd " + this->workspace_dir;
    std::string compile_cmd =
        change_dir_cmd + "; " + this->project->compile_cmd;
    ret_code = WEXITSTATUS(std::system(compile_cmd.c_str()));
    ADT_CHECK(ret_code == 0) << adt::errors::RuntimeError{
        std::string() + "system() failed. ret_code: " +
        std::to_string(ret_code) + ", compile_cmd: " + compile_cmd};
    return adt::Ok{};
  }

  std::string GetSoPath() {
    return this->workspace_dir + "/" + this->project->so_relative_path;
  }

 private:
  std::string workspace_dir;
  Project project;

  adt::Result<adt::Ok> DumpNestedFilesToFs(
      const Directory<File>& directory, const std::string& relative_dir_path) {
    std::string dir_path = this->workspace_dir + "/" + relative_dir_path;
    std::string cmd = std::string() + "mkdir -p " + dir_path;
    ADT_CHECK(WEXITSTATUS(std::system(cmd.c_str())) == 0);
    using Ok = adt::Result<adt::Ok>;
    for (const auto& [dentry, file] : directory.dentry2file->storage) {
      ADT_RETURN_IF_ERR(file.Match(
          [&](const FileContent& file_content) -> Ok {
            return DumpFileContentToFs(file_content,
                                       relative_dir_path + "/" + dentry);
          },
          [&](const SoftLink& soft_link) -> Ok {
            return DumpSoftLinkToFs(soft_link,
                                    relative_dir_path + "/" + dentry);
          },
          [&](const Directory<File>& sub_dir) -> Ok {
            return DumpNestedFilesToFs(sub_dir,
                                       relative_dir_path + "/" + dentry);
          }));
    }
    return adt::Ok{};
  }

  adt::Result<adt::Ok> DumpFileContentToFs(
      const FileContent& file_content, const std::string& relative_file_path) {
    std::string file_path = this->workspace_dir + "/" + relative_file_path;
    std::ofstream of{file_path};
    ADT_CHECK(of.is_open()) << adt::errors::RuntimeError{
        std::string() + "file open failed. file_path: " + file_path};
    of << file_content->file_content;
    of.close();
    return adt::Ok{};
  }

  adt::Result<adt::Ok> DumpSoftLinkToFs(const SoftLink& soft_link,
                                        const std::string& relative_link_path) {
    std::string link = this->workspace_dir + "/" + relative_link_path;
    std::optional<std::string> target_path;
    auto FindExistedSourcePath =
        [&](const auto& prefix) -> adt::Result<adt::LoopCtrl> {
      std::string cur_target_path =
          std::string() + prefix + "/" + soft_link->target_relative_path;
      if (FileExists(cur_target_path)) {
        target_path = cur_target_path;
        return adt::Break{};
      } else {
        return adt::Continue{};
      }
    };
    ADT_RETURN_IF_ERR(env::VisitEachApPath(FindExistedSourcePath));
    ADT_CHECK(target_path.has_value()) << adt::errors::RuntimeError{
        std::string() +
        "link failed. relative_path: " + soft_link->target_relative_path};
    std::string cmd =
        std::string() + "ln -s " + target_path.value() + " " + link;
    ADT_CHECK(WEXITSTATUS(std::system(cmd.c_str())) == 0);
    return adt::Ok{};
  }

  bool FileExists(const std::string& filepath) {
    std::fstream fp;
    fp.open(filepath, std::fstream::in);
    if (fp.is_open()) {
      fp.close();
      return true;
    } else {
      return false;
    }
  }
};

}  // namespace ap::code_module
