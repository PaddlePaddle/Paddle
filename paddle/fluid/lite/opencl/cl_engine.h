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

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/lite/opencl/cl_include.h"
#include "paddle/fluid/lite/opencl/cl_tool.h"

namespace paddle {
namespace lite {

class CLEngine {
 public:
  static CLEngine* Global();

  bool Init();

  cl::Platform& platform();

  cl::Context& context();

  cl::Device& device();

  cl::CommandQueue& command_queue();

  std::unique_ptr<cl::Program> CreateProgram(const cl::Context& context,
                                             std::string file_name);

  std::unique_ptr<cl::UserEvent> CreateEvent(const cl::Context& context);

  bool BuildProgram(cl::Program* program, const std::string& options = "");

  bool IsInitSuccess() { return is_init_success_; }

  std::string cl_path() { return cl_path_; }

  void set_cl_path(std::string cl_path) { cl_path_ = cl_path; }

 private:
  CLEngine() = default;

  ~CLEngine();

  bool InitializePlatform();

  bool InitializeDevice();

  std::shared_ptr<cl::Context> CreateContext() {
    auto context = std::make_shared<cl::Context>(
        std::vector<cl::Device>{device()}, nullptr, nullptr, nullptr, &status_);
    CL_CHECK_ERRORS(status_);
    return context;
  }

  std::shared_ptr<cl::CommandQueue> CreateCommandQueue(
      const cl::Context& context) {
    auto queue =
        std::make_shared<cl::CommandQueue>(context, device(), 0, &status_);
    CL_CHECK_ERRORS(status_);
    return queue;
  }

  std::string cl_path_;

  std::shared_ptr<cl::Platform> platform_{nullptr};

  std::shared_ptr<cl::Context> context_{nullptr};

  std::shared_ptr<cl::Device> device_{nullptr};

  std::shared_ptr<cl::CommandQueue> command_queue_{nullptr};

  cl_int status_{CL_SUCCESS};

  bool initialized_{false};

  bool is_init_success_{false};
};

}  // namespace lite
}  // namespace paddle
