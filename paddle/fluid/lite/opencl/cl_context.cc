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

#include <glog/logging.h>
#include <memory>
#include <string>
#include <utility>

#include "paddle/fluid/lite/opencl/cl_context.h"
#include "paddle/fluid/lite/opencl/cl_engine.h"
#include "paddle/fluid/lite/opencl/cl_tool.h"

namespace paddle {
namespace lite {

cl::CommandQueue &CLContext::GetCommandQueue() {
  return CLEngine::Global()->command_queue();
}

cl::Context &CLContext::GetContext() { return CLEngine::Global()->context(); }

cl::Program &CLContext::GetProgram(const std::string &file_name,
                                   const std::string &options) {
  std::string program_key = file_name;
  if (!options.empty()) {
    program_key += options;
  }
  auto it = programs_.find(program_key);
  if (it != programs_.end()) {
    VLOG(3) << " --- program -> " << program_key << " has been built --- ";
    return *(it->second);
  }

  auto program = CLEngine::Global()->CreateProgram(
      GetContext(), CLEngine::Global()->cl_path() + "/cl_kernel/" + file_name);

  VLOG(3) << " --- begin build program -> " << program_key << " --- ";
  CLEngine::Global()->BuildProgram(program.get(), options);
  VLOG(3) << " --- end build program -> " << program_key << " --- ";

  programs_[program_key] = std::move(program);

  return *(programs_[program_key]);
}

std::unique_ptr<cl::Kernel> CLContext::GetKernel(const std::string &kernel_name,
                                                 const std::string &file_name,
                                                 const std::string &options) {
  cl_int status{CL_SUCCESS};
  VLOG(3) << " --- to get program " << file_name << " --- ";
  auto program = GetProgram(file_name, options);
  VLOG(3) << " --- end get program --- ";
  VLOG(3) << " --- to create kernel: " << kernel_name << " --- ";
  std::unique_ptr<cl::Kernel> kernel(
      new cl::Kernel(program, kernel_name.c_str(), &status));
  CL_CHECK_ERRORS(status);
  VLOG(3) << " --- end create kernel --- ";
  return std::move(kernel);
}

}  // namespace lite
}  // namespace paddle
