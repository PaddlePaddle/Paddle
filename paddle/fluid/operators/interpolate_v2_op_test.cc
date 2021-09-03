// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"

USE_OP(nearest_interp_v2);
USE_OP(nearest_interp_v2_grad);
USE_OP(bilinear_interp_v2);
USE_OP(bilinear_interp_v2_grad);

#ifdef PADDLE_WITH_ASCEND_CL
USE_OP_DEVICE_KERNEL(nearest_interp_v2, NPU);
USE_OP_DEVICE_KERNEL(nearest_interp_v2_grad, NPU);
USE_OP_DEVICE_KERNEL(bilinear_interp_v2, NPU);
USE_OP_DEVICE_KERNEL(bilinear_interp_v2_grad, NPU);
#endif

namespace paddle {
namespace operators {

template <typename T>
static void print_data(const T* data, const int64_t& numel,
                       const std::string& name) {
  printf("%s = [ ", name.c_str());
  for (int64_t i = 0; i < numel; ++i) {
    if (std::is_floating_point<T>::value) {
      printf("%.1f, ", static_cast<float>(data[i]));
    } else {
      printf("%d, ", static_cast<int>(data[i]));
    }
  }
  printf("]\n");
}

template <typename T>
static void print_data(const T* data,
                       const std::vector<int64_t>& dims,
                       const std::string& name) {
  const int64_t numel = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
  int64_t stride1 = dims[dims.size() - 1];
  int64_t stride2 = stride1 * dims[dims.size() - 2];
  int64_t index = 0;
  printf("\n%s = \n[ ", name.c_str());
  while (index < numel) {
    if (std::is_floating_point<T>::value) {
      printf("%.1f ", static_cast<float>(data[index]));
    } else {
      printf("%d ", static_cast<int>(data[index]));
    }
    if ((index + 1) % stride1 == 0) printf("]\n[ ");
    if ((index + 1) % stride2 == 0) printf("\n[ ");
    index++;
  }
}

template <typename T>
static void feed_value(const platform::DeviceContext& ctx,
                       const framework::DDim dims, framework::LoDTensor* tensor,
                       const T value) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = static_cast<T>(value);
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
}

template <typename T>
static void feed_range(const platform::DeviceContext& ctx,
                       const framework::DDim dims, framework::LoDTensor* tensor,
                       const T value) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = static_cast<T>(value + i);
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
}

const std::string interp_method = "bilinear";
// const std::string interp_method = "bilinear";
const std::string data_layout = "NCHW";
const bool align_corners = false;
const int64_t batch_size = 2;
const int64_t input_c = 3;
const int64_t input_h = 5;
const int64_t input_w = 7;
const std::vector<int64_t> x_dims = {batch_size, input_c, input_h, input_w};
const int64_t x_numel = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int64_t>());

const int64_t out_h = 7;
const int64_t out_w = 10;
const std::vector<int64_t> out_dims = {batch_size, input_c, out_h, out_w};
const int64_t out_numel = std::accumulate(out_dims.begin(), out_dims.end(), 1, std::multiplies<int64_t>());

template <typename T>
void TestMain(const platform::DeviceContext& ctx, std::vector<T>* out_data, std::vector<T>* x_grad_data) {
  auto place = ctx.GetPlace();

  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim x_ddims = framework::make_ddim(x_dims);
  // framework::DDim size_ddims = framework::make_ddim({1});
  framework::DDim out_ddims = framework::make_ddim(out_dims);

  // --------------- forward ----------------------
  desc_fwd.SetType("bilinear_interp_v2");
  desc_fwd.SetInput("X", {"X"});
  // desc_fwd.SetInput("SizeTensor", {"x0", "x1"});
  // desc_fwd.SetInput("OutSize", {"OutSize"});
  desc_fwd.SetOutput("Out", {"Out"});
  desc_fwd.SetAttr("data_layout", data_layout);
  desc_fwd.SetAttr("interp_method", interp_method);
  desc_fwd.SetAttr("align_corners", align_corners);
  desc_fwd.SetAttr("align_mode", 1);
  desc_fwd.SetAttr("out_d", 1);
  desc_fwd.SetAttr("out_h", 60);
  desc_fwd.SetAttr("out_w", 25);
  desc_fwd.SetAttr("scale", std::vector<float>{1.5, 1.5});

  auto x_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  // auto out_size_tensor = scope.Var("OutSize")->GetMutable<framework::LoDTensor>();
  // auto size_h_tensor = scope.Var("x0")->GetMutable<framework::LoDTensor>();
  // auto size_w_tensor = scope.Var("x1")->GetMutable<framework::LoDTensor>();
  auto out_tensor = scope.Var("Out")->GetMutable<framework::LoDTensor>();

  feed_range<T>(ctx, x_ddims, x_tensor, static_cast<T>(1.0));
  // feed_value<T>(ctx, x_ddims, x_tensor, static_cast<T>(1.0));
  // feed_value<int32_t>(ctx, framework::make_ddim({2}), out_size_tensor, static_cast<int32_t>(out_h));
  // framework::TensorFromVector(std::vector<int32_t>({3, 8}), ctx, out_size_tensor);
  // feed_value<int32_t>(ctx, size_ddims, size_h_tensor, static_cast<int32_t>(3));
  // feed_value<int32_t>(ctx, size_ddims, size_w_tensor, static_cast<int32_t>(3));

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*out_tensor, ctx, out_data);

  // --------------- backward ----------------------
  desc_bwd.SetType("bilinear_interp_v2_grad");
  desc_bwd.SetInput("X", {"X"});
  // desc_bwd.SetInput("SizeTensor", {"x0", "x1"});
  // desc_bwd.SetInput("OutSize", {"OutSize"});
  desc_bwd.SetInput(framework::GradVarName("Out"), {framework::GradVarName("Out")});
  desc_bwd.SetOutput(framework::GradVarName("X"), {framework::GradVarName("X")});
  desc_bwd.SetAttr("data_layout", data_layout);
  desc_bwd.SetAttr("interp_method", interp_method);
  desc_bwd.SetAttr("align_corners", align_corners);
  desc_bwd.SetAttr("align_mode", 1);
  desc_bwd.SetAttr("out_d", 1);
  desc_bwd.SetAttr("out_h", 60);
  desc_bwd.SetAttr("out_w", 25);
  desc_bwd.SetAttr("scale", std::vector<float>{1.5, 1.5});
  desc_bwd.SetAttr("use_mkldnn", false);

  auto out_grad_tensor = scope.Var(framework::GradVarName("Out"))->GetMutable<framework::LoDTensor>();
  auto x_grad_tensor = scope.Var(framework::GradVarName("X"))->GetMutable<framework::LoDTensor>();
  feed_value<T>(ctx, out_ddims, out_grad_tensor, static_cast<T>(1.0));

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*x_grad_tensor, ctx, x_grad_data);
}

template <typename T>
static void compare_results(const std::vector<T>& cpu_data,
                            const std::vector<T>& dev_data,
                            const std::vector<int64_t>& dims,
                            const std::string& name) {
  auto result = std::equal(
      cpu_data.begin(), cpu_data.end(), dev_data.begin(),
      [](const float& l, const float& r) { return fabs(l - r) < 1e-9; });
  if (!result) {
    LOG(INFO) << "=========== " << name << " is NOT Equal !!!!!!!!! ===========";
    print_data(cpu_data.data(), dims, name + "_cpu");
    print_data(dev_data.data(), dims, name + "_dev");
  } else {
    LOG(INFO) << "=========== " << name << " is Equal in CPU and GPU ===========";
    // print_data(cpu_data.data(), dims, name + "_cpu");
    // print_data(dev_data.data(), dims, name + "_dev");
  }
}

TEST(test_interpolate_op, compare_cpu_and_dev) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> cpu_out_data(out_numel);
  std::vector<float> cpu_grad_data(x_numel);
  TestMain<float>(cpu_ctx, &cpu_out_data, &cpu_grad_data);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  platform::CUDAPlace gpu_place;
  platform::CUDADeviceContext gpu_ctx(gpu_place);
  std::vector<float> gpu_out_data(out_numel);
  std::vector<float> gpu_grad_data(x_numel);
  TestMain<float>(gpu_ctx, &gpu_out_data, &gpu_grad_data);

  compare_results<float>(cpu_out_data, gpu_out_data, out_dims, "output");
  compare_results<float>(cpu_grad_data, gpu_grad_data, x_dims, "x_grad");
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  platform::NPUPlace npu_place(0);
  platform::NPUDeviceContext npu_ctx(npu_place);
  std::vector<float> npu_out_data(out_numel);
  std::vector<float> npu_grad_data(x_numel);
  TestMain<float>(npu_ctx, &npu_out_data, &npu_grad_data);

  compare_results<float>(cpu_out_data, npu_out_data, out_dims, "output");
  compare_results<float>(cpu_grad_data, npu_grad_data, x_dims, "x_grad");
#endif
}

}  // namespace operators
}  // namespace paddle