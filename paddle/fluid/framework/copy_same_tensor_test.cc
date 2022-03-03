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

#include <sys/types.h>
#include <random>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/ddim.h"

DECLARE_bool(use_system_allocator);

namespace paddle {
namespace framework {

static std::vector<platform::Place> CreatePlaceList() {
  std::vector<platform::Place> places;
  places.emplace_back(platform::CPUPlace());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  places.emplace_back(platform::CUDAPlace(0));
#endif
  return places;
}

template <typename T>
static bool CopySameTensorTestMain(const DDim &dims,
                                   const platform::Place &src_place,
                                   const platform::Place &dst_place,
                                   bool sync_copy) {
  FLAGS_use_system_allocator = true;  // force to use system allocator

  // Step 1: create a cpu tensor and initialize it with random value;
  Tensor src_cpu_tensor;
  {
    src_cpu_tensor.Resize(dims);
    auto *src_ptr_cpu = src_cpu_tensor.mutable_data<T>(platform::CPUPlace());
    int64_t num = src_cpu_tensor.numel();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(-1000, 1000);
    for (int64_t i = 0; i < num; ++i) {
      src_ptr_cpu[i] = dist(gen);
    }
  }

  // Step 2: copy the source tensor to dst place
  Tensor dst_cpu_tensor;
  {
    Tensor src_tensor;
    TensorCopySync(src_cpu_tensor, src_place, &src_tensor);

    // The source tensor and dst_tensor is the same
    if (sync_copy) {
      TensorCopySync(src_tensor, dst_place, &src_tensor);
    } else {
      paddle::framework::TensorCopy(src_tensor, dst_place, &src_tensor);
      platform::DeviceContextPool::Instance().Get(src_place)->Wait();
      platform::DeviceContextPool::Instance().Get(dst_place)->Wait();
    }

    // Get the result cpu tensor
    TensorCopySync(src_tensor, platform::CPUPlace(), &dst_cpu_tensor);
  }

  const void *ground_truth_ptr = src_cpu_tensor.data();
  const void *result_ptr = dst_cpu_tensor.data();
  size_t byte_num = phi::product(dims) * sizeof(T);
  return std::memcmp(ground_truth_ptr, result_ptr, byte_num) == 0;
}

TEST(test_tensor_copy, test_copy_same_tensor) {
  using DataType = float;
  auto dims = phi::make_ddim({3, 4, 5});

  auto places = CreatePlaceList();
  for (auto &src_p : places) {
    for (auto &dst_p : places) {
      ASSERT_TRUE(CopySameTensorTestMain<DataType>(dims, src_p, dst_p, true));
      ASSERT_TRUE(CopySameTensorTestMain<DataType>(dims, src_p, dst_p, false));
    }
  }
}

}  // namespace framework
}  // namespace paddle
