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

USE_OP(matmul_v2);
USE_OP(matmul_v2_grad);

#ifdef PADDLE_WITH_ASCEND_CL
USE_OP_DEVICE_KERNEL(matmul_v2, NPU);
USE_OP_DEVICE_KERNEL(matmul_v2_grad, NPU);
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
  // print_data<T>(data.data(), framework::vectorize(dims), "feed_range");
  framework::TensorFromVector(data, ctx, tensor);
  tensor->Resize(dims);
}

// Case 1: [K] x [K] = [1]

// Case 2: X=(M, K), Y=(K, N) => Out=(M, N)
//         [M, K] x [K, N] = [M, N]
//            [K] x [K, N] =    [N]
//         [M, K] x [K]    = [M]
// Case 3: X=(B, M, K), Y=(K, N) => Out=(B, M, N) && trans_x = false
//         [2, M, K] x [K, N] =    [2, M, N]
//      [2, 2, M, K] x [K, N] = [2, 2, M, N]
//      [2, 1, M, K] x [K, N] = [2, 1, M, N]
//      [1, 2, M, K] x [K, N] = [1, 2, M, N]
// Case 4:
//         [2, M, K] x    [2, K, N] = [2, M, N]
//         [1, M, K] x    [2, K, N] = [2, M, N]
//            [M, K] x    [2, K, N] = [2, M, N]
//               [K] x    [2, K, N] = [2,    N]
//      [1, 2, M, K] x    [2, K, N] = [1, 2, M, N]
//      [2, 1, M, K] x [1, 2, K, N] = [2, 2, M, N]
//      [1, 1, M, K] x [1, 2, K, N] = [1, 2, M, N]
//         [2, M, K] x [2, 2, K, N] = [2, 2, M, N]
//            [M, K] x [2, 2, K, N] = [2, 2, M, N]
//               [K] x [2, 2, K, N] = [2, 2,    N]
//      [1, 2, M, K] x [2, 1, K, N] = [2, 2, M, N]
//      [2, 1, M, K] x [1, 2, K, N] = [2, 2, M, N]
//      [2, 2, M, K] x [1, 1, K, N] = [2, 2, M, N]
//      [2, 2, M, K] x    [2, K, N] = [2, 2, M, N]
//      [2, 2, M, K] x       [K, N] = [2, 2, M, N]
//      [2, 2, M, K] x       [K]    = [2, 2, M]

// Attrs
const bool trans_x = true;
const bool trans_y = false;
// Shape
const int64_t M = 2;
const int64_t K = 3;
const int64_t N = 4;
// input - X
const std::vector<int64_t> x_dim = trans_x ? std::vector<int64_t>{1, 1, 100, 1} : std::vector<int64_t>{1, 1, 1, 100};
const int64_t x_numel = std::accumulate(x_dim.begin(), x_dim.end(), 1, std::multiplies<int64_t>());
// input - Y
const std::vector<int64_t> y_dim = trans_y ? std::vector<int64_t>{100} : std::vector<int64_t>{100};
const int64_t y_numel = std::accumulate(y_dim.begin(), y_dim.end(), 1, std::multiplies<int64_t>());
// Output - Out
const std::vector<int64_t> out_dim = {1, 1, 1};
const int64_t out_numel = std::accumulate(out_dim.begin(), out_dim.end(), 1, std::multiplies<int64_t>());

template <typename T>
void TestMain(const platform::DeviceContext& ctx, std::vector<T>* out_data, std::vector<T>* x_grad_data, std::vector<T>* y_grad_data) {
  auto place = ctx.GetPlace();

  framework::Scope scope;
  framework::OpDesc desc_fwd;
  framework::OpDesc desc_bwd;

  framework::DDim x_ddim = framework::make_ddim(x_dim);
  framework::DDim y_ddim = framework::make_ddim(y_dim);
  framework::DDim out_ddim = framework::make_ddim(out_dim);

  LOG(INFO) << "x_ddim=" << x_ddim.to_str();
  LOG(INFO) << "y_ddim=" << y_ddim.to_str();
  LOG(INFO) << "out_ddim=" << out_ddim.to_str();

  // --------------- forward ----------------------
  desc_fwd.SetType("matmul_v2");
  desc_fwd.SetInput("X", {"X"});
  desc_fwd.SetInput("Y", {"Y"});
  desc_fwd.SetOutput("Out", {"Out"});
  desc_fwd.SetAttr("trans_x", trans_x);
  desc_fwd.SetAttr("trans_y", trans_y);
  // desc_fwd.SetAttr("use_cudnn", false);
  // desc_fwd.SetAttr("use_mkldnn", false);

  auto X = scope.Var("X")->GetMutable<framework::LoDTensor>();
  auto Y = scope.Var("Y")->GetMutable<framework::LoDTensor>();
  auto Out = scope.Var("Out")->GetMutable<framework::LoDTensor>();

  feed_range<T>(ctx, x_ddim, X, static_cast<T>(0.0));
  feed_range<T>(ctx, y_ddim, Y, static_cast<T>(0.0));

  auto op_fwd = framework::OpRegistry::CreateOp(desc_fwd);

  LOG(INFO) << op_fwd->DebugStringEx(&scope);
  op_fwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_fwd->DebugStringEx(&scope);

  framework::TensorToVector(*Out, ctx, out_data);

  // // --------------- backward ----------------------
  desc_bwd.SetType("matmul_v2_grad");
  desc_bwd.SetInput("X", {"X"});
  desc_bwd.SetInput("Y", {"Y"});
  desc_bwd.SetInput(framework::GradVarName("Out"), {framework::GradVarName("Out")});
  desc_bwd.SetOutput(framework::GradVarName("X"), {framework::GradVarName("X")});
  desc_bwd.SetOutput(framework::GradVarName("Y"), {framework::GradVarName("Y")});
  desc_bwd.SetAttr("trans_x", trans_x);
  desc_bwd.SetAttr("trans_y", trans_y);
  // desc_bwd.SetAttr("use_cudnn", false);
  desc_bwd.SetAttr("use_mkldnn", false);

  auto dOut = scope.Var(framework::GradVarName("Out"))->GetMutable<framework::LoDTensor>();
  auto dX   = scope.Var(framework::GradVarName("X"))->GetMutable<framework::LoDTensor>();
  auto dY   = scope.Var(framework::GradVarName("Y"))->GetMutable<framework::LoDTensor>();

  feed_value<T>(ctx, out_ddim, dOut, static_cast<T>(1.0));

  auto op_bwd = framework::OpRegistry::CreateOp(desc_bwd);

  LOG(INFO) << op_bwd->DebugStringEx(&scope);
  op_bwd->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();
  LOG(INFO) << op_bwd->DebugStringEx(&scope);

  framework::TensorToVector(*dX, ctx, x_grad_data);
  framework::TensorToVector(*dY, ctx, y_grad_data);
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

TEST(test_matmul_v2, compare_cpu_and_npu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext cpu_ctx(cpu_place);
  std::vector<float> out_cpu(out_numel);
  std::vector<float> x_grad_cpu(x_numel);
  std::vector<float> y_grad_cpu(y_numel);
  TestMain<float>(cpu_ctx, &out_cpu, &x_grad_cpu, &y_grad_cpu);

#ifdef PADDLE_WITH_ASCEND_CL
  platform::NPUPlace npu_place(0);
  platform::NPUDeviceContext npu_ctx(npu_place);
  std::vector<float> out_npu(out_numel);
  std::vector<float> x_grad_npu(x_numel);
  std::vector<float> y_grad_npu(y_numel);
  TestMain<float>(npu_ctx, &out_npu, &x_grad_npu, &y_grad_npu);

  compare_results<float>(out_cpu, out_npu, out_dim, "Out");
  compare_results<float>(x_grad_cpu, x_grad_npu, x_dim, "x_grad");
  compare_results<float>(y_grad_cpu, y_grad_npu, y_dim, "y_grad");
#endif
}

}  // namespace operators
}  // namespace paddle
