// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <ATen/cuda/EmptyTensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>

#include "gtest/gtest.h"
#include "paddle/phi/api/include/torch_compat_runtime.h"
#include "paddle/phi/common/float16.h"

TEST(conversion_basic_test, BasicCase) {
  at::Tensor a =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  at::Tensor b = at::full({2, 3}, 2, at::kFloat);
  double c = 10;

  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < a_contig.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  // Show result
  for (int64_t i = 0; i < a_contig.numel(); i++) {
    std::cout << "Result[" << i << "] = " << a_ptr[i] * b_ptr[i] + c
              << std::endl;
    ASSERT_EQ(result_ptr[i], 12);
  }
  // for test empty_cuda:
  at::Tensor bb =
      at::detail::empty_cuda(12, at::kFloat, at::kCUDA, std::nullopt);

  // for test sizoof(at::Half):
  std::cout << sizeof(at::Half) << std::endl;
  at::Tensor num_non_exiting_ctas = at::empty(
      {}, at::TensorOptions().device(a.device()).dtype(at::ScalarType::Int));

  {
    std::vector<int64_t> shape = {2, 3, 4, 5};
    size_t size_ =
        c10::elementSize(at::ScalarType::Float) * c10::multiply_integers(shape);
    std::cout << "multiply_integers out: " << size_ << std::endl;
  }
  {
    std::vector<int> shape = {2, 3, 4, 5};
    size_t size_ =
        c10::elementSize(at::ScalarType::Float) * c10::sum_integers(shape);
    std::cout << "sum_integers out: " << size_ << std::endl;
  }
  {
    auto stream = at::cuda::getCurrentCUDAStream();
    std::cout << "stream num: " << stream.stream() << std::endl;
    at::cuda::stream_synchronize(stream);
    at::Tensor bb =
        at::detail::empty_cuda(12, at::kFloat, at::kCUDA, std::nullopt);
  }
  {
    at::Tensor a = at::ones(
        {2, 3}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    std::cout << "a.device() is at::kCUDA: " << (a.device().type() == at::kCUDA)
              << std::endl;
    const c10::cuda::CUDAGuard device_guard(a.device());
    std::cout << "device_guard is at::kCUDA: "
              << (device_guard.current_device().type() == at::kCUDA)
              << std::endl;
    const c10::cuda::OptionalCUDAGuard device_guard_opt(a.device());
    std::cout << "device_guard is at::kCUDA: "
              << (device_guard_opt.current_device().value().type() == at::kCUDA)
              << std::endl;
  }
}
