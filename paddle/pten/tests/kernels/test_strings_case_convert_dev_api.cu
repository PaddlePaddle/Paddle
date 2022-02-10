/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <stdio.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#if (defined(__NVCC__) || defined(__HIPCC__))
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#endif

#include "paddle/fluid/platform/device/gpu/gpu_helper.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/pstring.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/string_tensor.h"
#include "paddle/pten/kernels/strings/case_convert_kernel.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = pten::framework::DDim;
using pstring = ::pten::dtype::pstring;
using paddle::platform::CPUPlace;
using paddle::platform::CUDAPlace;
using paddle::platform::CUDADeviceContext;

__global__ void CopyFromVec(pstring* dst, char** src, const int64_t num) {
  CUDA_KERNEL_LOOP(i, num) { dst[i] = pstring(src[i]); }
}

__global__ void CopyToVec(char** dst, pstring** src, const int64_t num) {
  CUDA_KERNEL_LOOP(i, num) {
    // Copy from gpu to cpu
    memcpy(dst[i], src[i]->data(), src[i]->size() + 1);
  }
}

TEST(DEV_API, strings_cast_convert) {
  auto gpu0 = CUDAPlace();
  const size_t MAX_SEQ_LEN = 64;

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  pten::GPUContext* dev_ctx =
      reinterpret_cast<pten::GPUContext*>(pool.Get(gpu0));

  // 1. create tensor
  const DDim dims({1, 2});
  StringTensorMeta meta(dims);
  const auto string_allocator =
      std::make_unique<paddle::experimental::DefaultAllocator>(gpu0);
  const auto alloc = string_allocator.get();
  StringTensor dense_x(alloc, meta);

  std::string short_str = "A Short Pstring.";
  std::string long_str = "A Large Pstring Whose Length Is Longer Than 22.";

  char* str_arr[2];
  for (int i = 0; i < 2; ++i) {
    cudaMalloc(&str_arr[i], MAX_SEQ_LEN);
  }

  cudaMemcpy(str_arr[0],
             short_str.data(),
             short_str.length() + 1,
             cudaMemcpyHostToDevice);
  cudaMemcpy(str_arr[1],
             long_str.data(),
             long_str.length() + 1,
             cudaMemcpyHostToDevice);
  char** gpu_str_arr;
  cudaMalloc(&gpu_str_arr, 2 * sizeof(char*));
  cudaMemcpy(gpu_str_arr, str_arr, 2 * sizeof(char*), cudaMemcpyHostToDevice);

  pstring* dense_x_data = dense_x.mutable_data(gpu0);
  CopyFromVec<<<1, 32>>>(dense_x_data, gpu_str_arr, 2);
  // 2. get expected results
  std::string expected_results[] = {short_str, short_str, long_str, long_str};
  std::transform(short_str.begin(),
                 short_str.end(),
                 expected_results[0].begin(),
                 ::tolower);
  std::transform(short_str.begin(),
                 short_str.end(),
                 expected_results[1].begin(),
                 ::toupper);
  std::transform(
      long_str.begin(), long_str.end(), expected_results[2].begin(), ::tolower);
  std::transform(
      long_str.begin(), long_str.end(), expected_results[3].begin(), ::toupper);
  // 3. test API, ascii encoding
  auto dense_lower_out = pten::strings::StringLower(*dev_ctx, "", dense_x);
  auto dense_upper_out = pten::strings::StringUpper(*dev_ctx, "", dense_x);

  // 4. check results
  ASSERT_EQ(dense_lower_out.numel(), 2);
  ASSERT_EQ(dense_upper_out.numel(), 2);
  dense_lower_out.data()[0];
  pstring* result_strs[] = {dense_lower_out.mutable_data(gpu0),
                            dense_upper_out.mutable_data(gpu0),
                            dense_lower_out.mutable_data(gpu0) + 1,
                            dense_upper_out.mutable_data(gpu0) + 1};
  pstring** gpu_result_strs;
  cudaMalloc(&gpu_result_strs, 4 * sizeof(pstring*));
  cudaMemcpy(gpu_result_strs,
             result_strs,
             4 * sizeof(pstring*),
             cudaMemcpyHostToDevice);

  std::vector<char> cpu_results_vec[4];
  char* cpu_results[4];
  for (int i = 0; i < 4; ++i) {
    cpu_results_vec[i].resize(MAX_SEQ_LEN);
    cpu_results[i] = cpu_results_vec[i].data();
  }

  char* gpu_results[4];
  for (int i = 0; i < 4; ++i) {
    cudaMalloc(&gpu_results[i], MAX_SEQ_LEN);
  }
  char** gpu_results_str_arr;
  cudaMalloc(&gpu_results_str_arr, 4 * sizeof(char*));
  cudaMemcpy(gpu_results_str_arr,
             gpu_results,
             4 * sizeof(char*),
             cudaMemcpyHostToDevice);
  CopyToVec<<<1, 32>>>(gpu_results_str_arr, gpu_result_strs, 4);
  for (int i = 0; i < 4; ++i) {
    cudaMemcpy(
        cpu_results[i], gpu_results[i], MAX_SEQ_LEN, cudaMemcpyDeviceToHost);
  }
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(cpu_results[i], expected_results[i]);
  }
}

// TEST(DEV_API, strings_cast_convert_utf8) {
//   auto gpu0 = CUDAPlace();
//   // 1. create tensor
//   const DDim dims({1, 1});
//   StringTensorMeta meta(dims);

//   const auto string_allocator =
//       std::make_unique<paddle::experimental::DefaultAllocator>(gpu0);
//   const auto alloc = string_allocator.get();
//   StringTensor dense_x(alloc, meta);

//   std::string utf8_str = "óÓsscHloëË";

//   pstring* dense_x_data = dense_x.mutable_data();
//   dense_x_data[0] = utf8_str;

//   paddle::platform::DeviceContextPool& pool =
//       paddle::platform::DeviceContextPool::Instance();
//   pten::GPUContext* dev_ctx =
//   reinterpret_cast<pten::GPUContext*>(pool.Get(gpu0));

//   // 2. get expected results
//   std::string expected_results[] = {"óósschloëë", "ÓÓSSCHLOËË"};
//   std::string cpu_results[] = {"", ""};
//   for (int i = 0; i < 2; ++i) {
//       cpu_results[i] = std::string(expected_results[i].length(), 0);
//   }
//   // 3. test API, ascii encoding
//   auto dense_lower_out = pten::StringLower(
//       *dev_ctx,
//       "utf-8",
//       dense_x);
//   auto dense_upper_out = pten::StringUpper(
//       *dev_ctx,
//       "utf-8",
//       dense_x);

//   // 4. check results
//   ASSERT_EQ(dense_lower_out.numel(), 1);
//   ASSERT_EQ(dense_upper_out.numel(), 1);

//   paddle::memory::Copy(CPUPlace(), const_cast<char*>(cpu_results[0].data()),
//   gpu0, dense_lower_out.data()[0].data(), cpu_results[0].length(),
//   dev_ctx->stream());
//   paddle::memory::Copy(CPUPlace(), const_cast<char*>(cpu_results[1].data()),
//   gpu0, dense_upper_out.data()[0].data(), cpu_results[1].length(),
//   dev_ctx->stream());

//   for (int i = 0; i < 2; ++i) {
//       ASSERT_EQ(cpu_results[i], expected_results[i]);
//   }
// }

}  // namespace tests
}  // namespace pten
