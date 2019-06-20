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
#include "paddle/fluid/lite/opencl/cl2_header.h"
#include "paddle/fluid/lite/opencl/cl_context.h"
#include "paddle/fluid/lite/opencl/cl_image.h"

namespace paddle {
namespace lite {

class CLHelper {
 public:
  CLHelper() = default;

  explicit CLHelper(CLContext *context) : context_(context) {}

  void AddKernel(const std::string &kernel_name, const std::string &file_name,
                 const std::string &options = "");

  cl::Kernel &KernelAt(const int index);

  cl::CommandQueue &OpenCLCommandQueue();

  cl::Context &OpenCLContext();

  std::vector<size_t> DefaultWorkSize(const CLImage &image);

 private:
  CLContext *context_;
  std::vector<std::unique_ptr<cl::Kernel>> kernels;
};

}  // namespace lite
}  // namespace paddle
