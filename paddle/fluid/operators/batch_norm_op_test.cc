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

USE_OP(batch_norm);
USE_OP(batch_norm_grad);

#ifdef PADDLE_WITH_ASCEND_CL
USE_OP_DEVICE_KERNEL(batch_norm, NPU);
USE_OP_DEVICE_KERNEL(batch_norm_grad, NPU);
#endif

namespace paddle {
namespace operators {

template <typename T>
static void print_data(const T* data, const int64_t& numel,
                       const std::string& name) {
  printf("%s = [ ", name.c_str());
  for (int64_t i = 0; i < numel; ++i) {
    if (std::is_floating_point<T>::value) {
      printf("% 9.7f, ", static_cast<float>(data[i]));
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
  if (dims.size() < 3) {
    print_data<T>(data, numel, name);
    return;
  }
  int64_t stride1 = dims[dims.size() - 1];
  int64_t stride2 = stride1 * dims[dims.size() - 2];
  int64_t index = 0;
  printf("\n%s = \n[ ", name.c_str());
  while (index < numel) {
    if (std::is_floating_point<T>::value) {
      // total width 9, pad with [space] in the begin, 7 digits after .
      printf("% 9.7f ", static_cast<float>(data[index]));
    } else {
      printf("%d ", static_cast<int>(data[index]));
    }
    if ( (index + 1) == numel) {
      printf("]\n");
    } else if ((index + 1) % stride2 == 0) {
      printf("]\n\n[ ");
    } else if ((index + 1) % stride1 == 0) {
      printf("]\n[ ");
    }
    index++;
  }
}

template <typename T>
static void feed_value(const platform::DeviceContext& ctx,
                       const framework::DDim dims, 
                       framework::LoDTensor* tensor,
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
                       const framework::DDim dims, 
                       framework::LoDTensor* tensor,
                       const T value) {
  size_t numel = static_cast<size_t>(framework::product(dims));
  std::vector<T> data(numel);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = static_cast<T>(value + i);
  }
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
}

// Attrs
const std::string data_layout = "NHWC"; // NLC
const float epsilon = 1e-5;
float momentum = 0.9;
const bool is_test = false;
const bool use_global_stats = false;
const bool trainable_stats = false;
// Input
const int64_t batch_size = 1;
const int64_t input_c = 5;
// const int64_t input_l = 3;
const int64_t input_h = 3;
const int64_t input_w = 4;
// Input - X
// const std::vector<int64_t> x_dims = {batch_size, input_c, input_h, input_w}; // NCHW
const std::vector<int64_t> x_dims = {batch_size, input_h, input_w, input_c}; // NHWC
// const std::vector<int64_t> x_dims = {batch_size, input_c, input_l}; //  NCL
// const std::vector<int64_t> x_dims = {batch_size, input_l, input_c}; //  NLC
const std::vector<int64_t> c_dims = {input_c};
const int64_t x_numel = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int64_t>());
// Input - Scale, Bias, Mean, Variance - {input_c}

template <typename T>
void TestMain(const platform::DeviceContext& ctx, 
              std::vector<T>* y_data,
              std::vector<T>* mean_out_data,
              std::vector<T>* var_out_data,
              std::vector<T>* saved_mean_data,
              std::vector<T>* saved_var_data,
              std::vector<T>* d_x_data,
              std::vector<T>* d_scale_data,
              std::vector<T>* d_bias_data) {
  auto place = ctx.GetPlace();

  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim x_ddims = framework::make_ddim(x_dims);
  framework::DDim c_ddims = framework::make_ddim({input_c});

  // --------------- forward ----------------------
  desc_fwd.SetType("batch_norm");
  desc_fwd.SetInput("X", {"X"});
  desc_fwd.SetInput("Scale", {"Scale"});
  desc_fwd.SetInput("Bias", {"Bias"});
  desc_fwd.SetInput("Mean", {"Mean"});
  desc_fwd.SetInput("Variance", {"Variance"});
  desc_fwd.SetOutput("Y", {"Y"});
  desc_fwd.SetOutput("MeanOut", {"Mean"}); // MeanOut share with Input Mean
  desc_fwd.SetOutput("VarianceOut", {"Variance"}); // VarianceOut share with Input Variance
  desc_fwd.SetOutput("SavedMean", {"SavedMean"});
  desc_fwd.SetOutput("SavedVariance", {"SavedVariance"});
  desc_fwd.SetOutput("ReserveSpace", {"ReserveSpace"});
  desc_fwd.SetAttr("data_layout", data_layout);
  desc_fwd.SetAttr("is_test", is_test);
  desc_fwd.SetAttr("use_global_stats", use_global_stats);
  desc_fwd.SetAttr("trainable_statistics", trainable_stats);
  desc_fwd.SetAttr("epsilon", epsilon);
  desc_fwd.SetAttr("momentum", momentum);

  // inputs
  auto x = scope.Var("X")->GetMutable<framework::LoDTensor>();
  auto scale = scope.Var("Scale")->GetMutable<framework::LoDTensor>();
  auto bias = scope.Var("Bias")->GetMutable<framework::LoDTensor>();
  auto mean = scope.Var("Mean")->GetMutable<framework::LoDTensor>();
  auto var = scope.Var("Variance")->GetMutable<framework::LoDTensor>();
  // outputs
  auto y = scope.Var("Y")->GetMutable<framework::LoDTensor>();
  auto mean_out = scope.Var("Mean")->GetMutable<framework::LoDTensor>(); // MeanOut share with Input Mean
  auto var_out = scope.Var("Variance")->GetMutable<framework::LoDTensor>(); // VarianceOut share with Input Variance
  auto saved_mean = scope.Var("SavedMean")->GetMutable<framework::LoDTensor>();
  auto saved_var = scope.Var("SavedVariance")->GetMutable<framework::LoDTensor>();
  scope.Var("ReserveSpace")->GetMutable<framework::LoDTensor>();

  // feed value to inputs
  feed_range<T>(ctx, x_ddims, x,     static_cast<T>(0.0));
  feed_value<T>(ctx, c_ddims, scale, static_cast<T>(1.0));
  feed_value<T>(ctx, c_ddims, bias,  static_cast<T>(0.0));
  feed_value<T>(ctx, c_ddims, mean,  static_cast<T>(0.0));
  feed_value<T>(ctx, c_ddims, var,   static_cast<T>(1.0));

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*y, ctx, y_data);
  framework::TensorToVector(*mean_out, ctx, mean_out_data);
  framework::TensorToVector(*var_out, ctx, var_out_data);
  framework::TensorToVector(*saved_mean, ctx, saved_mean_data);
  framework::TensorToVector(*saved_var, ctx, saved_var_data);

  // --------------- backward ----------------------
  desc_bwd.SetType("batch_norm_grad");
  desc_bwd.SetInput("X", {"X"});
  desc_bwd.SetInput("Scale", {"Scale"});
  desc_bwd.SetInput("Bias", {"Bias"});
  desc_bwd.SetInput(framework::GradVarName("Y"), {framework::GradVarName("Y")});
  desc_bwd.SetInput("SavedMean", {"SavedMean"});
  desc_bwd.SetInput("SavedVariance", {"SavedVariance"});
  desc_bwd.SetOutput(framework::GradVarName("X"), {framework::GradVarName("X")});
  desc_bwd.SetOutput(framework::GradVarName("Scale"), {framework::GradVarName("Scale")});
  desc_bwd.SetOutput(framework::GradVarName("Bias"), {framework::GradVarName("Bias")});
  desc_bwd.SetAttr("data_layout", data_layout);
  desc_bwd.SetAttr("is_test", is_test);
  desc_bwd.SetAttr("use_global_stats", use_global_stats);
  desc_bwd.SetAttr("trainable_statistics", trainable_stats);
  desc_bwd.SetAttr("epsilon", epsilon);
  desc_bwd.SetAttr("momentum", momentum);
  desc_bwd.SetAttr("use_mkldnn", false);
  desc_bwd.SetAttr("fuse_with_relu", false);

  // new inputs
  auto y_grad = scope.Var(framework::GradVarName("Y"))->GetMutable<framework::LoDTensor>();
  feed_value<T>(ctx, x_ddims, y_grad, static_cast<T>(1.0));
  // outputs
  auto d_x = scope.Var(framework::GradVarName("X"))->GetMutable<framework::LoDTensor>();
  auto d_scale = scope.Var(framework::GradVarName("Scale"))->GetMutable<framework::LoDTensor>();
  auto d_bias = scope.Var(framework::GradVarName("Bias"))->GetMutable<framework::LoDTensor>();

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*d_x, ctx, d_x_data);
  framework::TensorToVector(*d_scale, ctx, d_scale_data);
  framework::TensorToVector(*d_bias, ctx, d_bias_data);
}

template <typename T>
static void compare_results(const std::vector<T>& cpu_data,
                            const std::vector<T>& dev_data,
                            const std::vector<int64_t>& dims,
                            const std::string& name) {
  auto result = std::equal(
      cpu_data.begin(), cpu_data.end(), dev_data.begin(),
      [](const float& l, const float& r) { return fabs(l - r) < 1e-5; });
  if (!result) {
    LOG(INFO) << "=========== " << name << " is NOT Equal !!!!!!!!! ===========";
    print_data(cpu_data.data(), dims, name + "_cpu");
    print_data(dev_data.data(), dims, name + "_dev");
  } else {
    LOG(INFO) << "=========== " << name << " is Equal in CPU and GPU ===========";
    print_data(cpu_data.data(), dims, name + "_cpu");
    print_data(dev_data.data(), dims, name + "_dev");
  }
}

TEST(test_interpolate_op, compare_cpu_and_dev) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> cpu_y(x_numel);
  std::vector<float> cpu_mean_out(x_numel);
  std::vector<float> cpu_var_out(input_c);
  std::vector<float> cpu_saved_mean(input_c);
  std::vector<float> cpu_saved_var(input_c);
  std::vector<float> cpu_d_x(x_numel);
  std::vector<float> cpu_d_scale(input_c);
  std::vector<float> cpu_d_bias(input_c);
  TestMain<float>(cpu_ctx, &cpu_y, &cpu_mean_out, &cpu_var_out, &cpu_saved_mean, &cpu_saved_var, &cpu_d_x, &cpu_d_scale, &cpu_d_bias);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  platform::CUDAPlace gpu_place;
  platform::CUDADeviceContext gpu_ctx(gpu_place);
  std::vector<float> gpu_y(x_numel);
  std::vector<float> gpu_mean_out(x_numel);
  std::vector<float> gpu_var_out(input_c);
  std::vector<float> gpu_saved_mean(input_c);
  std::vector<float> gpu_saved_var(input_c);
  std::vector<float> gpu_d_x(x_numel);
  std::vector<float> gpu_d_scale(input_c);
  std::vector<float> gpu_d_bias(input_c);
  TestMain<float>(gpu_ctx, &gpu_y, &gpu_mean_out, &gpu_var_out, &gpu_saved_mean, &gpu_saved_var, &gpu_d_x, &gpu_d_scale, &gpu_d_bias);

  compare_results<float>(cpu_y, gpu_y, x_dims, "y");
  compare_results<float>(cpu_mean_out, gpu_mean_out, c_dims, "mean_out");
  compare_results<float>(cpu_var_out, gpu_var_out, c_dims, "var_out");
  compare_results<float>(cpu_saved_mean, gpu_saved_mean, c_dims, "saved_mean");
  compare_results<float>(cpu_saved_var, gpu_saved_var, c_dims, "saved_var");
  compare_results<float>(cpu_d_x, gpu_d_x, x_dims, "x_grad");
  compare_results<float>(cpu_d_scale, gpu_d_scale, c_dims, "scale_grad");
  compare_results<float>(cpu_d_bias, gpu_d_bias, c_dims, "bias_grad");
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  platform::NPUPlace npu_place(0);
  platform::NPUDeviceContext npu_ctx(npu_place);
  std::vector<float> npu_y(x_numel);
  std::vector<float> npu_mean_out(x_numel);
  std::vector<float> npu_var_out(input_c);
  std::vector<float> npu_saved_mean(input_c);
  std::vector<float> npu_saved_var(input_c);
  std::vector<float> npu_d_x(x_numel);
  std::vector<float> npu_d_scale(input_c);
  std::vector<float> npu_d_bias(input_c);
  TestMain<float>(npu_ctx, &npu_y, &npu_mean_out, &npu_var_out, &npu_saved_mean, &npu_saved_var, &npu_d_x, &npu_d_scale, &npu_d_bias);

  compare_results<float>(cpu_y, npu_y, x_dims, "y");
  compare_results<float>(cpu_mean_out, npu_mean_out, c_dims, "mean_out");
  compare_results<float>(cpu_var_out, npu_var_out, c_dims, "var_out");
  compare_results<float>(cpu_saved_mean, npu_saved_mean, c_dims, "saved_mean");
  compare_results<float>(cpu_saved_var, npu_saved_var, c_dims, "saved_var");
  compare_results<float>(cpu_d_x, npu_d_x, x_dims, "x_grad");
  compare_results<float>(cpu_d_scale, npu_d_scale, c_dims, "scale_grad");
  compare_results<float>(cpu_d_bias, npu_d_bias, c_dims, "bias_grad");
#endif
}

}  // namespace operators
}  // namespace paddle