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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/opencl/cl_caller.h"
#include "paddle/fluid/lite/opencl/cl_context.h"
#include "paddle/fluid/lite/opencl/cl_engine.h"
#include "paddle/fluid/lite/opencl/cl_helper.h"
#include "paddle/fluid/lite/opencl/cl_image.h"

DEFINE_string(cl_path, "/data/local/tmp/opencl", "The OpenCL kernels path.");

namespace paddle {
namespace lite {

TEST(cl_test, engine_test) {
  auto* engine = CLEngine::Global();
  CHECK(engine->IsInitSuccess());
  engine->set_cl_path(FLAGS_cl_path);
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
  engine->set_cl_path(FLAGS_cl_path);
  CLContext context;
  context.GetKernel("pool_max", "pool_kernel.cl", "");
  context.GetKernel("elementwise_add", "elementwise_add_kernel.cl", "");
  context.GetKernel("elementwise_add", "elementwise_add_kernel.cl", "");
}

TEST(cl_test, kernel_test) {
  auto* engine = CLEngine::Global();
  CHECK(engine->IsInitSuccess());
  engine->set_cl_path(FLAGS_cl_path);
  std::unique_ptr<CLContext> context(new CLContext);
  // std::unique_ptr<CLHelper> helper(new CLHelper(context.get()));
  std::unique_ptr<CLHelper> helper(new CLHelper);
  helper->set_context(context.get());
  helper->AddKernel("elementwise_add", "elementwise_add_kernel.cl");
  helper->AddKernel("pool_max", "pool_kernel.cl");
  helper->AddKernel("elementwise_add", "elementwise_add_kernel.cl");
  auto kernel = helper->GetKernel(2);

  std::unique_ptr<float[]> in_data(new float[4 * 3 * 256 * 512]);
  for (int i = 0; i < 4 * 3 * 256 * 512; i++) {
    in_data[i] = 1.f;
  }
  const DDim in_dim = DDim(std::vector<DDim::value_type>{4, 3, 256, 512});
  CLImage in_image;
  in_image.set_tensor_data(in_data.get(), in_dim);
  in_image.InitNormalCLImage(helper->OpenCLContext());
  LOG(INFO) << in_image;

  std::unique_ptr<float[]> bias_data(new float[4 * 3 * 256 * 512]);
  for (int i = 0; i < 4 * 3 * 256 * 512; i++) {
    bias_data[i] = 2.f;
  }
  const DDim bias_dim = DDim(std::vector<DDim::value_type>{4, 3, 256, 512});
  CLImage bias_image;
  bias_image.set_tensor_data(bias_data.get(), bias_dim);
  bias_image.InitNormalCLImage(helper->OpenCLContext());
  LOG(INFO) << bias_image;

  CLImage out_image;
  const DDim out_dim = DDim(std::vector<DDim::value_type>{4, 3, 256, 512});
  out_image.InitEmptyImage(helper->OpenCLContext(), out_dim);
  LOG(INFO) << out_image;

  cl_int status;
  status = kernel.setArg(0, *in_image.cl_image());
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(1, *bias_image.cl_image());
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(2, *out_image.cl_image());
  CL_CHECK_ERRORS(status);

  // auto global_work_size = helper->DefaultWorkSize(out_image);
  size_t width = in_image.ImageWidth();
  size_t height = in_image.ImageHeight();
  auto global_work_size = cl::NDRange{width, height};
  cl::Event event;
  status = helper->OpenCLCommandQueue().enqueueNDRangeKernel(
      kernel, cl::NullRange, global_work_size, cl::NullRange, nullptr, &event);
  CL_CHECK_ERRORS(status);
  status = helper->OpenCLCommandQueue().finish();
  CL_CHECK_ERRORS(status);
  double start_nanos = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  double stop_nanos = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  double elapsed_micros = (stop_nanos - start_nanos) / 1000.0;
  LOG(INFO) << "Kernel Run Cost Time: " << elapsed_micros << " us.";
  LOG(INFO) << out_image;
}

TEST(cl_test, channel_add_test) {
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);

  const DDim in_dim = DDim(std::vector<DDim::value_type>{4, 16, 256, 512});
  std::unique_ptr<float[]> in_data(new float[4 * 16 * 256 * 512]);
  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    in_data[i] = dist(engine);
  }

  const DDim bias_dim = DDim(std::vector<DDim::value_type>{16});
  std::unique_ptr<float[]> bias_data(new float[16]);
  for (int i = 0; i < 16; i++) {
    bias_data[i] = dist(engine);
  }

  std::unique_ptr<float[]> out_ref(new float[4 * 16 * 256 * 512]);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 16; j++) {
      float b = bias_data[j];
      for (int k = 0; k < 256 * 512; k++) {
        int index = (i * 16 + j) * 256 * 512 + k;
        out_ref[index] = in_data[index] + b;
      }
    }
  }

  const DDim out_dim = DDim(std::vector<DDim::value_type>{4, 16, 256, 512});
  std::unique_ptr<float[]> out(new float[4 * 16 * 256 * 512]);

  bool status = InitOpenCLEngine(FLAGS_cl_path);
  CHECK(status) << "Fail to initialize OpenCL engine.";
  std::unique_ptr<CLContext> context(new CLContext);
  std::unique_ptr<CLHelper> helper(new CLHelper(context.get()));
  helper->AddKernel("elementwise_add", "elementwise_add_kernel.cl");
  helper->AddKernel("channel_add", "channel_add_kernel.cl");
  elementwise_add(helper.get(), in_data.get(), in_dim, bias_data.get(),
                  bias_dim, out.get(), out_dim);

  int stride = 4 * 16 * 256 * 512 / 20;
  for (int i = 0; i < 4 * 16 * 256 * 512; i += stride) {
    std::cout << out[i] << " ";
  }

  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    EXPECT_NEAR(out[i], out_ref[i], 1e-6);
  }

  std::cout << std::endl;
}

TEST(cl_test, elementwise_add_test) {
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);

  const DDim in_dim = DDim(std::vector<DDim::value_type>{4, 16, 256, 512});
  std::unique_ptr<float[]> in_data(new float[4 * 16 * 256 * 512]);
  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    in_data[i] = dist(engine);
  }

  const DDim bias_dim = DDim(std::vector<DDim::value_type>{4, 16, 256, 512});
  std::unique_ptr<float[]> bias_data(new float[4 * 16 * 256 * 512]);
  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    bias_data[i] = dist(engine);
  }

  std::unique_ptr<float[]> out_ref(new float[4 * 16 * 256 * 512]);
  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    out_ref[i] = in_data[i] + bias_data[i];
  }

  const DDim out_dim = DDim(std::vector<DDim::value_type>{4, 16, 256, 512});
  std::unique_ptr<float[]> out(new float[4 * 16 * 256 * 512]);

  bool status = InitOpenCLEngine(FLAGS_cl_path);
  CHECK(status) << "Fail to initialize OpenCL engine.";
  std::unique_ptr<CLContext> context(new CLContext);
  std::unique_ptr<CLHelper> helper(new CLHelper(context.get()));
  helper->AddKernel("elementwise_add", "elementwise_add_kernel.cl");
  helper->AddKernel("channel_add", "channel_add_kernel.cl");
  elementwise_add(helper.get(), in_data.get(), in_dim, bias_data.get(),
                  bias_dim, out.get(), out_dim);

  int stride = 4 * 16 * 256 * 512 / 20;
  for (int i = 0; i < 4 * 16 * 256 * 512; i += stride) {
    std::cout << out[i] << " ";
  }

  for (int i = 0; i < 4 * 16 * 256 * 512; i++) {
    EXPECT_NEAR(out[i], out_ref[i], 1e-6);
  }

  std::cout << std::endl;
}

}  // namespace lite
}  // namespace paddle
