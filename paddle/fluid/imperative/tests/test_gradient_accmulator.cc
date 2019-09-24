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

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/fluid/memory/memcpy.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;
namespace framework = paddle::framework;
namespace paddle {
namespace imperative {

void TensorAdd(const framework::Variable& src, framework::Variable* dst);

#if defined(PADDLE_WITH_CUDA)
template <typename T>
int TensorGPUAddTest(platform::CUDAPlace place, T t1, T t2) {
  framework::Variable var1;
  framework::Variable var2;
  std::vector<T> src_data(10, t1);
  std::vector<T> dst_data(10, t2);
  std::vector<T> result;
  platform::CPUPlace src_place;
  for (unsigned int i = 0; i < 10; i++) {
    result.emplace_back(src_data[i] + dst_data[i]);
  }
  std::vector<int64_t> dims = {2, 5};
  auto* src = var1.GetMutable<framework::LoDTensor>();
  auto* dst = var2.GetMutable<framework::LoDTensor>();
  src->Resize(framework::make_ddim(dims));
  dst->Resize(framework::make_ddim(dims));
  auto* src_mutable = src->mutable_data<T>(place);
  auto* dst_mutable = dst->mutable_data<T>(place);
  paddle::memory::Copy(place, src_mutable, src_place, src_data.data(),
                       sizeof(T) * src_data.size(), 0);
  paddle::memory::Copy(place, dst_mutable, src_place, dst_data.data(),
                       sizeof(T) * dst_data.size(), 0);
  imperative::TensorAdd(var1, &var2);
  framework::LoDTensor rlt;
  platform::CPUPlace rlt_place;
  framework::TensorCopySync(*dst, rlt_place, &rlt);

  for (unsigned int i = 0; i < rlt.numel(); i++) {
    if (rlt.data<T>()[i] != result[i]) return 1;
  }
  return 0;
}
#endif

template <typename T>
int TensorCPUAddTest(platform::CPUPlace place, T t1, T t2) {
  framework::Variable var1;
  framework::Variable var2;
  std::vector<T> src_data(10, t1);
  std::vector<T> dst_data(10, t2);
  std::vector<T> result;
  platform::CPUPlace src_place;
  for (unsigned int i = 0; i < 10; i++) {
    result.emplace_back(src_data[i] + dst_data[i]);
  }
  std::vector<int64_t> dims = {2, 5};
  auto* src = var1.GetMutable<framework::LoDTensor>();
  auto* dst = var2.GetMutable<framework::LoDTensor>();
  src->Resize(framework::make_ddim(dims));
  dst->Resize(framework::make_ddim(dims));
  auto* src_mutable = src->mutable_data<T>(place);
  auto* dst_mutable = dst->mutable_data<T>(place);
  paddle::memory::Copy(place, src_mutable, src_place, src_data.data(),
                       sizeof(T) * src_data.size());
  paddle::memory::Copy(place, dst_mutable, src_place, dst_data.data(),
                       sizeof(T) * dst_data.size());
  imperative::TensorAdd(var1, &var2);
  framework::LoDTensor rlt;
  platform::CPUPlace rlt_place;
  framework::TensorCopySync(*dst, rlt_place, &rlt);

  for (unsigned int i = 0; i < rlt.numel(); i++) {
    if (rlt.data<T>()[i] != result[i]) return 1;
  }
  return 0;
}

TEST(test_add_functor, add_functor) {
#if defined(PADDLE_WITH_CUDA)
  platform::CUDAPlace gpu_place(0);
#endif
  platform::CPUPlace cpu_place;

  int cpu_res = 1;
  cpu_res = TensorCPUAddTest(cpu_place, 1.0, 0.0);
  EXPECT_EQ(cpu_res, 0);
  cpu_res = TensorCPUAddTest(cpu_place, static_cast<double>(1.0),
                             static_cast<double>(2.0));
  EXPECT_EQ(cpu_res, 0);
#if defined(PADDLE_WITH_CUDA)
  int gpu_res = 1;
  gpu_res = TensorGPUAddTest(gpu_place, 1.0, 0.0);
  EXPECT_EQ(gpu_res, 0);
  gpu_res = TensorGPUAddTest(gpu_place, static_cast<double>(1.0),
                             static_cast<double>(2.0));
  EXPECT_EQ(gpu_res, 0);
#endif
}

}  // namespace imperative
}  // namespace paddle
