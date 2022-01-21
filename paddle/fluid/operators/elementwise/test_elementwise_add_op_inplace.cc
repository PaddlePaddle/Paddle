// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <random>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

USE_OP(elementwise_add);

namespace paddle {
namespace operators {

static void Memcpy(void *dst, const void *src, size_t n, bool copy_to_gpu) {
  if (copy_to_gpu) {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
#elif defined(PADDLE_WITH_HIP)
    PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
#else
    PADDLE_THROW(
        platform::errors::InvalidArgument("Check your paddle version, current "
                                          "version is not compiled with cuda"));
#endif
  } else {
    std::memcpy(dst, src, n);
  }
}

template <typename T>
bool TestMain(const platform::Place &place, const framework::DDim &dims,
              bool inplace) {
  framework::Scope scope;
  auto *x = scope.Var("x")->GetMutable<framework::LoDTensor>();
  auto *y = scope.Var("y")->GetMutable<framework::LoDTensor>();
  auto *z = scope.Var("z")->GetMutable<framework::LoDTensor>();

  x->Resize(dims);
  y->Resize(dims);
  z->Resize(dims);

  size_t numel = static_cast<size_t>(framework::product(dims));

  auto x_ptr = x->mutable_data<T>(place);
  auto y_ptr = y->mutable_data<T>(place);
  auto z_ptr = z->mutable_data<T>(place);

  std::uniform_real_distribution<T> dist(static_cast<T>(10.0),
                                         static_cast<T>(20.0));
  std::mt19937 engine;
  std::vector<T> x_data(numel), y_data(numel), z_data(numel);
  std::vector<T> sum_result(numel);

  for (size_t i = 0; i < numel; ++i) {
    x_data[i] = dist(engine);
    y_data[i] = dist(engine);
    sum_result[i] = x_data[i] + y_data[i];
    z_data[i] = -1.0;  // set some data that is not existed
  }

  auto bytes = sizeof(T) * numel;
  bool is_gpu_place = platform::is_gpu_place(place);
  Memcpy(x_ptr, x_data.data(), bytes, is_gpu_place);
  Memcpy(y_ptr, y_data.data(), bytes, is_gpu_place);
  Memcpy(z_ptr, z_data.data(), bytes, is_gpu_place);

  const char *out_name = inplace ? "x" : "z";
  auto op = framework::OpRegistry::CreateOp("elementwise_add",
                                            {{"X", {"x"}}, {"Y", {"y"}}},
                                            {{"Out", {out_name}}}, {});
  op->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();

  framework::LoDTensor cpu_out;
  auto &out_tensor = scope.FindVar(out_name)->Get<framework::LoDTensor>();
  PADDLE_ENFORCE_EQ(scope.kids().empty(), true,
                    platform::errors::InvalidArgument(
                        "The scope can not have the child scopes,"
                        "please check your code."));
  if (inplace) {
    PADDLE_ENFORCE_EQ(
        &out_tensor, x,
        platform::errors::InvalidArgument(
            "The output tensor should be same as input x in inplace mode,"
            " but now is not same."));
  } else {
    PADDLE_ENFORCE_EQ(
        &out_tensor, z,
        platform::errors::InvalidArgument(
            "The output tensor should be same as output z in normal mode,"
            " but now is not same."));
  }

  if (is_gpu_place) {
    framework::TensorCopySync(out_tensor, platform::CPUPlace(), &cpu_out);
  } else {
    cpu_out = out_tensor;
  }

  auto *out_ptr = cpu_out.data<T>();
  bool is_equal = std::equal(out_ptr, out_ptr + numel, sum_result.data());
  return is_equal;
}

TEST(test_elementwise_add_inplace, cpu_place) {
  framework::DDim dims({32, 64});
  platform::CPUPlace p;
  ASSERT_TRUE(TestMain<float>(p, dims, true));
}

TEST(test_elementwise_add_not_inplace, cpu_place) {
  framework::DDim dims({32, 64});
  platform::CPUPlace p;
  ASSERT_TRUE(TestMain<float>(p, dims, false));
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(test_elementwise_add_inplace, gpu_place) {
  framework::DDim dims({32, 64});
  platform::CUDAPlace p(0);
  ASSERT_TRUE(TestMain<float>(p, dims, true));
}

TEST(test_elementwise_add_not_inplace, gpu_place) {
  framework::DDim dims({32, 64});
  platform::CUDAPlace p(0);
  ASSERT_TRUE(TestMain<float>(p, dims, false));
}
#endif

}  // namespace operators
}  // namespace paddle
