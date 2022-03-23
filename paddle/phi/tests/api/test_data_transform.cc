/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include <memory>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, CPU, ALL_LAYOUT);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(full, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(copy, GPU, ALL_LAYOUT);
#endif

namespace paddle {
namespace tests {

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, data_transform_same_place) {
  // 1. create tensor
  auto x = paddle::experimental::full({3, 3},
                                      1.0,
                                      experimental::DataType::COMPLEX128,
                                      experimental::CPUPlace());

  auto y = paddle::experimental::full(
      {3, 3}, 2.0, experimental::DataType::FLOAT32, experimental::CPUPlace());

  std::vector<phi::dtype::complex<double>> sum(9, 6.0);

  // 2. test API
  auto out = paddle::experimental::matmul(x, y, false, false);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);
  ASSERT_EQ(out.type(), phi::DataType::COMPLEX128);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());

  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(sum[i].real,
                dense_out->data<phi::dtype::complex<double>>()[i].real,
                1e-6f);
    ASSERT_NEAR(sum[i].imag,
                dense_out->data<phi::dtype::complex<double>>()[i].imag,
                1e-6f);
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(Tensor, data_transform_diff_place) {
  // 1. create tensor
  auto x = paddle::experimental::full(
      {3, 3}, 1.0, experimental::DataType::FLOAT64, experimental::CPUPlace());

  auto y = paddle::experimental::full(
      {3, 3}, 2.0, experimental::DataType::FLOAT64, experimental::GPUPlace());

  std::vector<float> sum(9, 6.0);

  // 2. test API
  auto out = paddle::experimental::matmul(x, y, false, false);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 3);
  ASSERT_EQ(out.numel(), 9);
  ASSERT_EQ(out.dtype(), phi::DataType::FLOAT64);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);
  ASSERT_EQ(out.impl()->place(),
            phi::TransToPhiPlace(experimental::Backend::GPU));

  auto ref_out = experimental::copy_to(out, experimental::CPUPlace(), true);

  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(ref_out.impl());
  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(sum[i], dense_out->data<double>()[i], 1e-6f);
  }
}

#endif

}  // namespace tests
}  // namespace paddle
