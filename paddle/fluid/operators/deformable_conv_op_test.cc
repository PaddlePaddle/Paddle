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

USE_OP(deformable_conv);
USE_OP(deformable_conv_grad);

#ifdef PADDLE_WITH_ASCEND_CL
USE_OP_DEVICE_KERNEL(deformable_conv, NPU);
USE_OP_DEVICE_KERNEL(deformable_conv_grad, NPU);
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
      printf("% 9.2f ", static_cast<float>(data[index]));
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

// input - shape
const int64_t batch_size = 2;
const int64_t input_channel = 1;
const int64_t input_height = 3;
const int64_t input_width = 3;
const int64_t kernel_h = 2;
const int64_t kernel_w = 2;
// filter -shape
const int64_t out_channel = 2;
const int groups = 1;
const int deformable_groups = 1;
const int im2col_step = 1;
// attrs
const std::vector<int> strides = {1, 1};
const std::vector<int> paddings = {0, 0};
const std::vector<int> dilations = {1, 1};
// output - shape
const int64_t out_height = (input_height + 2 * paddings[0] - (dilations[0] * (kernel_h - 1) + 1)) / strides[0] + 1;
const int64_t out_width = (input_width + 2 * paddings[1] - (dilations[1] * (kernel_w - 1) + 1)) / strides[1] + 1;

// input - N, C_in, H_in, W_in
const std::vector<int64_t> input_dim_vec = {batch_size, input_channel, input_height, input_width}; // NCHW
const int64_t input_numel = std::accumulate(input_dim_vec.begin(), input_dim_vec.end(), 1, std::multiplies<int64_t>());
// filter - C_out, C_in // groups, Kh, Kw
const std::vector<int64_t> filter_dim_vec = {out_channel,  input_channel / groups, kernel_h, kernel_w};
const int64_t filter_numel = std::accumulate(filter_dim_vec.begin(), filter_dim_vec.end(), 1, std::multiplies<int64_t>());
// offset - N, 2 * Kh * Kw, H_out, W_out
const std::vector<int64_t> offset_dim_vec = {batch_size,  2 * kernel_h * kernel_w, out_height, out_width};
const int64_t offset_numel = std::accumulate(offset_dim_vec.begin(), offset_dim_vec.end(), 1, std::multiplies<int64_t>());
// mask - N, Kh * Kw, H_out, W_out
const std::vector<int64_t> mask_dim_vec = {batch_size,  kernel_h * kernel_w, out_height, out_width};
const int64_t mask_numel = std::accumulate(mask_dim_vec.begin(), mask_dim_vec.end(), 1, std::multiplies<int64_t>());
// output
const std::vector<int64_t> out_dim_vec = {batch_size, out_channel, out_height, out_width};  // NHWC
const int64_t out_numel = std::accumulate(out_dim_vec.begin(), out_dim_vec.end(), 1, std::multiplies<int64_t>());

template <typename T>
void TestMain(const platform::DeviceContext& ctx,
              std::vector<T>* output_data,
              std::vector<T>* input_grad_data,
              std::vector<T>* filter_grad_data,
              std::vector<T>* offset_grad_data,
              std::vector<T>* mask_grad_data) {
  auto place = ctx.GetPlace();

  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim input_dims = framework::make_ddim(input_dim_vec);
  framework::DDim filter_dims = framework::make_ddim(filter_dim_vec);
  framework::DDim offset_dims = framework::make_ddim(offset_dim_vec);
  framework::DDim mask_dims = framework::make_ddim(mask_dim_vec);
  framework::DDim out_dims = framework::make_ddim(out_dim_vec);

  LOG(INFO) << "input_dims=" << input_dims.to_str();
  LOG(INFO) << "filter_dims=" << filter_dims.to_str();
  LOG(INFO) << "offset_dims=" << offset_dims.to_str();
  LOG(INFO) << "mask_dims=" << mask_dims.to_str();
  LOG(INFO) << "out_dims=" << out_dims.to_str();

  // --------------- forward ----------------------
  desc_fwd.SetType("deformable_conv");
  desc_fwd.SetInput("Input", {"Input"});
  desc_fwd.SetInput("Filter", {"Filter"});
  desc_fwd.SetInput("Offset", {"Offset"});
  desc_fwd.SetInput("Mask", {"Mask"});
  desc_fwd.SetOutput("Output", {"Output"});
#ifdef PADDLE_WITH_ASCEND_CL
  desc_fwd.SetOutput("OffsetOut", {"OffsetOut"});
#endif
  desc_fwd.SetAttr("groups", groups);
  desc_fwd.SetAttr("deformable_groups", deformable_groups);
  desc_fwd.SetAttr("im2col_step", im2col_step);
  desc_fwd.SetAttr("strides", strides);
  desc_fwd.SetAttr("paddings", paddings);
  desc_fwd.SetAttr("dilations", dilations);

  auto input_tensor = scope.Var("Input")->GetMutable<framework::LoDTensor>();
  auto filter_tensor = scope.Var("Filter")->GetMutable<framework::LoDTensor>();
  auto offset_tensor = scope.Var("Offset")->GetMutable<framework::LoDTensor>();
  auto mask_tensor = scope.Var("Mask")->GetMutable<framework::LoDTensor>();
  auto output_tensor = scope.Var("Output")->GetMutable<framework::LoDTensor>();
#ifdef PADDLE_WITH_ASCEND_CL
  scope.Var("OffsetOut")->GetMutable<framework::LoDTensor>();
#endif

  // feed value to inputs
  feed_range<T>(ctx, input_dims,  input_tensor,   static_cast<T>(0.0));
  feed_value<T>(ctx, filter_dims, filter_tensor,  static_cast<T>(1.0));
  feed_value<T>(ctx, offset_dims, offset_tensor,  static_cast<T>(1.0));
  feed_value<T>(ctx, mask_dims,   mask_tensor,    static_cast<T>(2.0));

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*output_tensor, ctx, output_data);

  // --------------- backward ----------------------
  desc_bwd.SetType("deformable_conv_grad");
  desc_bwd.SetInput("Input", {"Input"});
  desc_bwd.SetInput("Filter", {"Filter"});
  desc_bwd.SetInput("Offset", {"Offset"});
  desc_bwd.SetInput("Mask", {"Mask"});
#ifdef PADDLE_WITH_ASCEND_CL
  desc_bwd.SetInput("OffsetOut", {"OffsetOut"});
#endif
  desc_bwd.SetInput(framework::GradVarName("Output"), {framework::GradVarName("Output")});
  desc_bwd.SetOutput(framework::GradVarName("Input"), {framework::GradVarName("Input")});
  desc_bwd.SetOutput(framework::GradVarName("Filter"), {framework::GradVarName("Filter")});
  desc_bwd.SetOutput(framework::GradVarName("Offset"), {framework::GradVarName("Offset")});
  desc_bwd.SetOutput(framework::GradVarName("Mask"), {framework::GradVarName("Mask")});
  desc_bwd.SetAttr("groups", groups);
  desc_bwd.SetAttr("deformable_groups", deformable_groups);
  desc_bwd.SetAttr("im2col_step", im2col_step);
  desc_bwd.SetAttr("strides", strides);
  desc_bwd.SetAttr("paddings", paddings);
  desc_bwd.SetAttr("dilations", dilations);

  auto out_grad_tensor = scope.Var(framework::GradVarName("Output"))->GetMutable<framework::LoDTensor>();
  auto input_grad_tensor = scope.Var(framework::GradVarName("Input"))->GetMutable<framework::LoDTensor>();
  auto filter_grad_tensor = scope.Var(framework::GradVarName("Filter"))->GetMutable<framework::LoDTensor>();
  auto offset_grad_tensor = scope.Var(framework::GradVarName("Offset"))->GetMutable<framework::LoDTensor>();
  auto mask_grad_tensor = scope.Var(framework::GradVarName("Mask"))->GetMutable<framework::LoDTensor>();

  // feed data to input tensors
  feed_value<T>(ctx, out_dims, out_grad_tensor,  static_cast<T>(1.0));

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*input_grad_tensor, ctx, input_grad_data);
  framework::TensorToVector(*filter_grad_tensor, ctx, filter_grad_data);
  framework::TensorToVector(*offset_grad_tensor, ctx, offset_grad_data);
  framework::TensorToVector(*mask_grad_tensor, ctx, mask_grad_data);
}

template <typename T>
static void compare_results(const std::vector<T>& cpu_data,
                            const std::vector<T>& dev_data,
                            const std::vector<int64_t>& dims,
                            const std::string& name) {
  auto result = std::equal(
      cpu_data.begin(), cpu_data.end(), dev_data.begin(),
      [](const float& l, const float& r) { return fabs(l - r) < 1e-2; });
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

TEST(test_pool2d_op, compare_cpu_and_gpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> output_cpu(out_numel);
  std::vector<float> input_grad_cpu(input_numel);
  std::vector<float> filter_grad_cpu(filter_numel);
  std::vector<float> offset_grad_cpu(offset_numel);
  std::vector<float> mask_grad_cpu(mask_numel);
  TestMain<float>(cpu_ctx, &output_cpu, &input_grad_cpu, &filter_grad_cpu, &offset_grad_cpu, &mask_grad_cpu);

#ifdef PADDLE_WITH_ASCEND_CL
  platform::NPUPlace npu_place(0);
  platform::NPUDeviceContext npu_ctx(npu_place);
  std::vector<float> output_npu(out_numel);
  std::vector<float> input_grad_npu(input_numel);
  std::vector<float> filter_grad_npu(filter_numel);
  std::vector<float> offset_grad_npu(offset_numel);
  std::vector<float> mask_grad_npu(mask_numel);
  TestMain<float>(npu_ctx, &output_npu, &input_grad_npu, &filter_grad_npu, &offset_grad_npu, &mask_grad_npu);

  compare_results<float>(output_cpu, output_npu, out_dim_vec, "output");
  compare_results<float>(input_grad_cpu, input_grad_npu, input_dim_vec, "input_grad");
  compare_results<float>(filter_grad_cpu, filter_grad_npu, filter_dim_vec, "filter_grad");
  compare_results<float>(offset_grad_cpu, offset_grad_npu, offset_dim_vec, "offset_grad");
  compare_results<float>(mask_grad_cpu, mask_grad_npu, mask_dim_vec, "mask_grad");
#endif
}

}  // namespace operators
}  // namespace paddle
