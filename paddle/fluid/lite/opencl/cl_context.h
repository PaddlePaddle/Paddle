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
#include <unordered_map>
#include "paddle/fluid/lite/opencl/cl_include.h"

namespace paddle {
namespace lite {

class CLContext {
 public:
  cl::CommandQueue &GetCommandQueue();

  cl::Context &GetContext();

  cl::Program &GetProgram(const std::string &file_name,
                          const std::string &options);

  std::unique_ptr<cl::Kernel> GetKernel(const std::string &kernel_name,
                                        const std::string &file_name,
                                        const std::string &options);

 private:
  std::unordered_map<std::string, std::unique_ptr<cl::Program>> programs_;
};

}  // namespace lite
}  // namespace paddle
