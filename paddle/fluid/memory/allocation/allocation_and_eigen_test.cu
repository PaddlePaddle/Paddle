// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/for_range.h"
#include "unsupported/Eigen/CXX11/Tensor"

// NOTE(yy): this unittest is not important. It just used for debugging.
// It can be removed later.
struct FillZero {
 public:
  float* ptr_;

  __device__ void operator()(size_t i) { ptr_[i] = 0.0f; }
};

namespace paddle {
TEST(Eigen, main) {
  framework::Tensor tensor;
  platform::CUDAPlace gpu(0);
  float* ptr = tensor.mutable_data<float>({10, 10}, gpu);
  auto& dev_ctx = *reinterpret_cast<platform::CUDADeviceContext*>(
      platform::DeviceContextPool::Instance().Get(gpu));
  PADDLE_ENFORCE(cudaMemset(ptr, 0, sizeof(float) * 100));

  platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx, 100);
  for_range(FillZero{ptr});
  dev_ctx.Wait();

  auto eigen_vec = framework::EigenVector<float>::Flatten(tensor);
  auto& eigen_dev = *dev_ctx.eigen_device();
  eigen_vec.device(eigen_dev) = eigen_vec.constant(0.0f);
}
}  // namespace paddle
