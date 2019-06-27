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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/lite/opencl/cl_context.h"
#include "paddle/fluid/lite/opencl/cl_image.h"
#include "paddle/fluid/lite/opencl/cl_include.h"

namespace paddle {
namespace lite {

class CLHelper {
 public:
  CLHelper() = default;

  explicit CLHelper(CLContext *context) : context_(context) {}

  void set_context(CLContext *context);

  void AddKernel(const std::string &kernel_name, const std::string &file_name,
                 const std::string &options = "");

  cl::Kernel &GetKernel(const int index);
  cl::Kernel &GetKernel(const std::string &name) {
    auto it = kernel_offset_.find(name);
    CHECK(it != kernel_offset_.end());
    return GetKernel(it->second);
  }

  cl::CommandQueue &OpenCLCommandQueue();

  cl::Context &OpenCLContext();

  cl::NDRange DefaultWorkSize(const CLImage &image);

 private:
  CLContext *context_{nullptr};
  std::map<std::string, int> kernel_offset_;
  std::vector<std::unique_ptr<cl::Kernel>> kernels_;
};

}  // namespace lite
}  // namespace paddle
