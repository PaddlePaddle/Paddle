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

USE_OP(conv2d_transpose);
USE_OP(conv2d_transpose_grad);

#ifdef PADDLE_WITH_ASCEND_CL
USE_OP_DEVICE_KERNEL(conv2d_transpose, NPU);
USE_OP_DEVICE_KERNEL(conv2d_transpose_grad, NPU);
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
static void print_data(const T* data, const std::vector<int64_t>& dims,
                       const std::string& name) {
  const int64_t numel =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
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
    if ((index + 1) == numel) {
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


// input - shape
const int64_t batch_size = 2;
const int64_t input_c = 1;
const int64_t input_h = 3;
const int64_t input_w = 3;
const int64_t kernel_h = 2;
const int64_t kernel_w = 2;
// filter -shape
const int64_t f_out_c = 2;
const int groups = 1;
const int deformable_groups = 1;
const int im2col_step = 1;
// attrs
const std::vector<int> strides = {1, 1};
const std::vector<int> paddings = {0, 0};
const std::vector<int> dilations = {1, 1};
const std::vector<int> output_padding = {0, 0};
const std::string padding_algorithm = "EXPLICIT";
// output - shape
const int64_t output_c = f_out_c * groups;
const int64_t d_bolck_h = dilations[0] * (kernel_h - 1) + 1;
const int64_t d_bolck_w = dilations[1] * (kernel_w - 1) + 1;
const int64_t output_h = (input_h - 1) * strides[0] + d_bolck_h;
const int64_t output_w = (input_w - 1) * strides[1] + d_bolck_w;

const std::string data_format = "NCHW";
// input - N, C_in, H_in, W_in
const std::vector<int64_t> input_dim = {batch_size, input_c, input_h, input_w}; // NCHW
const int64_t input_numel = std::accumulate(input_dim.begin(), input_dim.end(), 1, std::multiplies<int64_t>());
// filter - C_in, f_C_out, K_h, K_w
const std::vector<int64_t> filter_dim = {input_c, f_out_c, kernel_h, kernel_w};
const int64_t filter_numel = std::accumulate(filter_dim.begin(), filter_dim.end(), 1, std::multiplies<int64_t>());
// output - N, C_out, H_out, W_out
const std::vector<int64_t> output_dim = {batch_size, output_c, output_h, output_w};  // NHWC
const int64_t output_numel = std::accumulate(output_dim.begin(), output_dim.end(), 1, std::multiplies<int64_t>());

template <typename T>
void TestMain(const platform::DeviceContext& ctx, std::vector<T>* output_data,
              std::vector<T>* input_grad_data,
              std::vector<T>* filter_grad_data) {
  auto place = ctx.GetPlace();

  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim input_ddim = framework::make_ddim(input_dim);
  framework::DDim filter_ddim = framework::make_ddim(filter_dim);
  framework::DDim out_ddim = framework::make_ddim(output_dim);

  LOG(INFO) << "input_ddim=" << input_ddim.to_str();
  LOG(INFO) << "filter_ddim=" << filter_ddim.to_str();
  LOG(INFO) << "out_ddim=" << out_ddim.to_str();

  // --------------- forward ----------------------
  desc_fwd.SetType("conv2d_transpose");
  desc_fwd.SetInput("Input", {"Input"});
  desc_fwd.SetInput("Filter", {"Filter"});
  desc_fwd.SetOutput("Output", {"Output"});
  desc_fwd.SetAttr("groups", groups);
  desc_fwd.SetAttr("strides", strides);
  desc_fwd.SetAttr("paddings", paddings);
  desc_fwd.SetAttr("dilations", dilations);
  desc_fwd.SetAttr("output_padding", output_padding);
  desc_fwd.SetAttr("data_format", data_format);
  desc_fwd.SetAttr("padding_algorithm", padding_algorithm);
  
  auto input = scope.Var("Input")->GetMutable<framework::LoDTensor>();
  auto filter = scope.Var("Filter")->GetMutable<framework::LoDTensor>();
  auto output = scope.Var("Output")->GetMutable<framework::LoDTensor>();

  feed_range<T>(ctx, input_ddim, input, static_cast<T>(0.0));
  feed_range<T>(ctx, filter_ddim, filter, static_cast<T>(0.0));

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*output, ctx, output_data);

  // --------------- backward ----------------------
  desc_bwd.SetType("conv2d_transpose_grad");
  desc_bwd.SetInput("Input", {"Input"});
  desc_bwd.SetInput("Filter", {"Filter"});
  desc_bwd.SetInput(framework::GradVarName("Output"), {framework::GradVarName("Output")});
  desc_bwd.SetOutput(framework::GradVarName("Input"), {framework::GradVarName("Input")});
  desc_bwd.SetOutput(framework::GradVarName("Filter"), {framework::GradVarName("Filter")});
  desc_bwd.SetAttr("groups", groups);
  desc_bwd.SetAttr("strides", strides);
  desc_bwd.SetAttr("paddings", paddings);
  desc_bwd.SetAttr("dilations", dilations);
  desc_bwd.SetAttr("output_padding", output_padding);
  desc_bwd.SetAttr("data_format", data_format);
  desc_bwd.SetAttr("padding_algorithm", padding_algorithm);

  auto d_out = scope.Var(framework::GradVarName("Output"))->GetMutable<framework::LoDTensor>();
  auto d_input = scope.Var(framework::GradVarName("Input"))->GetMutable<framework::LoDTensor>();
  auto d_filter = scope.Var(framework::GradVarName("Filter"))->GetMutable<framework::LoDTensor>();

  feed_value<T>(ctx, out_ddim, d_out, static_cast<T>(1.0));

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*d_input, ctx, input_grad_data);
  framework::TensorToVector(*d_filter, ctx, filter_grad_data);
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
    LOG(INFO) << "=========== " << name
              << " is NOT Equal !!!!!!!!! ===========";
    print_data(cpu_data.data(), dims, name + "_cpu");
    print_data(dev_data.data(), dims, name + "_dev");
  } else {
    LOG(INFO) << "=========== " << name
              << " is Equal in CPU and DEV ===========";
    print_data(cpu_data.data(), dims, name + "_cpu");
    print_data(dev_data.data(), dims, name + "_dev");
  }
}

TEST(test_log_softmax_op, compare_cpu_and_npu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> output_cpu(output_numel);
  std::vector<float> d_input_cpu(input_numel);
  std::vector<float> d_filter_cpu(filter_numel);
  TestMain<float>(cpu_ctx, &output_cpu, &d_input_cpu, &d_filter_cpu);

#ifdef PADDLE_WITH_ASCEND_CL
  platform::NPUPlace npu_place(0);
  platform::NPUDeviceContext npu_ctx(npu_place);
  std::vector<float> output_npu(output_numel);
  std::vector<float> d_input_npu(input_numel);
  std::vector<float> d_filter_npu(filter_numel);
  TestMain<float>(npu_ctx, &output_npu, &d_input_npu, &d_filter_npu);

  compare_results<float>(output_cpu, output_npu, output_dim, "output");
  compare_results<float>(d_input_cpu, d_input_npu, input_dim, "input_grad");
  compare_results<float>(d_filter_cpu, d_filter_npu, filter_dim, "filter_grad");
#endif
}

}  // namespace operators
}  // namespace paddle