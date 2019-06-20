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
#include <gtest/gtest.h>
#include "paddle/fluid/lite/opencl/cl_context.h"
#include "paddle/fluid/lite/opencl/cl_engine.h"

namespace paddle {
namespace lite {

TEST(cl_test, engine_test) {
  auto* engine = CLEngine::Global();
  CHECK(engine->IsInitSuccess());
  engine->set_cl_path("/work/Develop/Paddle/paddle/fluid/lite/opencl");
  engine->platform();
  engine->device();
  engine->command_queue();
  auto& context = engine->context();
  auto program = engine->CreateProgram(
      context, engine->cl_path() + "/cl_kernel/" + "elementwise_add_kernel.cl");
  auto event = engine->CreateEvent(context);
  CHECK(engine->BuildProgram(program.get()));
}

TEST(cl_test, context_test) {
  auto* engine = CLEngine::Global();
  CHECK(engine->IsInitSuccess());
  engine->set_cl_path("/work/Develop/Paddle/paddle/fluid/lite/opencl");
  CLContext context;
  context.GetKernel("batchnorm", "batchnorm_kernel.cl", "");
  context.GetKernel("elementwise_add", "elementwise_add_kernel.cl", "");
  context.GetKernel("elementwise_add", "elementwise_add_kernel.cl", "");
}
}  // namespace lite
}  // namespace paddle
