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

#include <ATen/core/TensorBody.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda_guard.h>

#include "gtest/gtest.h"
#include "paddle/phi/api/include/torch_compat_runtime.h"
#include "paddle/phi/common/float16.h"

TEST(TensorBaseTest, DataPtrAPIs) {
  // Test data_ptr() and const_data_ptr() APIs
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);

  // Test void* data_ptr()
  void* void_ptr = tensor.data_ptr();
  ASSERT_NE(void_ptr, nullptr);

  // Test typed data_ptr<T>()
  float* float_ptr = tensor.data_ptr<float>();
  ASSERT_NE(float_ptr, nullptr);
  ASSERT_EQ(float_ptr, void_ptr);

  // Test const_data_ptr()
  const float* const_float_ptr = tensor.const_data_ptr<float>();
  ASSERT_NE(const_float_ptr, nullptr);
  ASSERT_EQ(const_float_ptr, float_ptr);

  // Test mutable_data_ptr()
  void* mutable_ptr = tensor.mutable_data_ptr();
  ASSERT_NE(mutable_ptr, nullptr);
  ASSERT_EQ(mutable_ptr, void_ptr);
}
TEST(TensorBaseTest, DimensionAPIs) {
  // Test dimension related APIs
  at::TensorBase tensor = at::ones({2, 3, 4}, at::kFloat);

  // Test sizes()
  auto sizes = tensor.sizes();
  ASSERT_EQ(sizes.size(), 3);
  ASSERT_EQ(sizes[0], 2);
  ASSERT_EQ(sizes[1], 3);
  ASSERT_EQ(sizes[2], 4);

  // Test size(dim)
  ASSERT_EQ(tensor.size(0), 2);
  ASSERT_EQ(tensor.size(1), 3);
  ASSERT_EQ(tensor.size(2), 4);

  // Test strides()
  auto strides = tensor.strides();
  ASSERT_EQ(strides.size(), 3);
  ASSERT_EQ(strides[0], 12);  // 3*4
  ASSERT_EQ(strides[1], 4);   // 4
  ASSERT_EQ(strides[2], 1);   // contiguous

  // Test stride(dim)
  ASSERT_EQ(tensor.stride(0), 12);
  ASSERT_EQ(tensor.stride(1), 4);
  ASSERT_EQ(tensor.stride(2), 1);

  // Test numel()
  ASSERT_EQ(tensor.numel(), 24);  // 2*3*4

  // Test dim()/ndimension()
  ASSERT_EQ(tensor.dim(), 3);
  ASSERT_EQ(tensor.ndimension(), 3);
}
TEST(TensorBaseTest, TypeDeviceAPIs) {
  // Test type and device related APIs
  at::TensorBase cpu_tensor = at::ones({2, 3}, at::kFloat);

  // Test dtype()/scalar_type()
  ASSERT_EQ(cpu_tensor.dtype(), at::kFloat);
  ASSERT_EQ(cpu_tensor.scalar_type(), at::kFloat);

  // Test device()
  ASSERT_EQ(cpu_tensor.device().type(), at::DeviceType::CPU);

  // Test get_device()
  ASSERT_EQ(cpu_tensor.get_device(), 0);  // CPU device index is -1

  // Test is_cpu()/is_cuda()
  ASSERT_TRUE(cpu_tensor.is_cpu());
  ASSERT_FALSE(cpu_tensor.is_cuda());

  // Test options()
  auto options = cpu_tensor.options();
  // ASSERT_EQ(options.dtype(), at::kFloat);
  ASSERT_EQ(options.device().type(), at::DeviceType::CPU);

  // Test with CUDA tensor if CUDA is available
  // if (at::cuda::is_available()) {
  // at::TensorBase cuda_tensor = at::ones({2, 3}, at::kFloat).cuda();

  // ASSERT_EQ(cuda_tensor.device().type(), at::DeviceType::CUDA);
  // ASSERT_GE(cuda_tensor.get_device(), 0);
  // ASSERT_FALSE(cuda_tensor.is_cpu());
  // ASSERT_TRUE(cuda_tensor.is_cuda());

  // auto cuda_options = cuda_tensor.options();
  // ASSERT_EQ(cuda_options.dtype(), at::kFloat);
  // ASSERT_EQ(cuda_options.device().type(), at::DeviceType::CUDA);
  // }
}

TEST(TensorBaseTest, ModifyOperationAPIs) {
  // Test modify operation related APIs
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);

  // Test is_contiguous()
  ASSERT_TRUE(tensor.is_contiguous());

  // Test fill_()
  tensor.fill_(2.0);
  float* data = tensor.data_ptr<float>();
  for (int i = 0; i < tensor.numel(); i++) {
    ASSERT_EQ(data[i], 2.0f);
  }

  // Test zero_()
  tensor.zero_();
  for (int i = 0; i < tensor.numel(); i++) {
    ASSERT_EQ(data[i], 0.0f);
  }

  // Test copy_()
  at::TensorBase src = at::ones({2, 3}, at::kFloat);
  tensor.copy_(src);
  for (int i = 0; i < tensor.numel(); i++) {
    ASSERT_EQ(data[i], 1.0f);
  }

  // Test view()
  at::TensorBase viewed = tensor.view({6});
  ASSERT_EQ(viewed.sizes(), std::vector<int64_t>{6});
  ASSERT_EQ(viewed.strides(), std::vector<int64_t>{1});
}

TEST(tensor_clone_test, BasicClone) {
  at::Tensor a = at::ones({2, 3}, at::kFloat);

  at::Tensor b = a.clone();

  ASSERT_EQ(a.sizes(), b.sizes());
  ASSERT_EQ(a.dtype(), b.dtype());
  ASSERT_EQ(a.device().type(), b.device().type());
}

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
    size_t size =
        c10::elementSize(at::ScalarType::Float) * c10::multiply_integers(shape);
    std::cout << "multiply_integers out: " << size << std::endl;
  }
  {
    std::vector<int> shape = {2, 3, 4, 5};
    size_t size =
        c10::elementSize(at::ScalarType::Float) * c10::sum_integers(shape);
    std::cout << "sum_integers out: " << size << std::endl;
  }
}
