/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/init.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace inference {

void Init(const std::vector<std::string> argv);

void LoadPersistables(framework::Executor* executor, framework::Scope* scope,
                      const framework::ProgramDesc& main_program,
                      const std::string& dirname,
                      const std::string& param_filename);

std::unique_ptr<framework::ProgramDesc> Load(framework::Executor* executor,
                                             framework::Scope* scope,
                                             const std::string& dirname);

std::unique_ptr<framework::ProgramDesc> Load(framework::Executor* executor,
                                             framework::Scope* scope,
                                             const std::string& prog_filename,
                                             const std::string& param_filename);

}  // namespace inference
}  // namespace paddle
