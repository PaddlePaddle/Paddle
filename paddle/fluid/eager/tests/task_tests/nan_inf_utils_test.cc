// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/nan_inf_utils.h"

#include <iostream>
#include <limits>
#include <tuple>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/strings_api.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(strings_empty, CPU, ALL_LAYOUT);

namespace egr {

#define CHECK_NAN_INF(tensors)                                               \
  {                                                                          \
    bool caught_exception = false;                                           \
    try {                                                                    \
      CheckTensorHasNanOrInf("nan_inf_test", tensors);                       \
    } catch (paddle::platform::EnforceNotMet & error) {                      \
      caught_exception = true;                                               \
      std::string ex_msg = error.what();                                     \
      EXPECT_TRUE(ex_msg.find("There are NAN or INF") != std::string::npos); \
    }                                                                        \
    EXPECT_TRUE(caught_exception);                                           \
  }

#define CHECK_NO_NAN_INF(tensors)                                            \
  {                                                                          \
    bool caught_exception = false;                                           \
    try {                                                                    \
      CheckTensorHasNanOrInf("nan_inf_test", tensors);                       \
    } catch (paddle::platform::EnforceNotMet & error) {                      \
      caught_exception = true;                                               \
      std::string ex_msg = error.what();                                     \
      EXPECT_TRUE(ex_msg.find("There are NAN or INF") != std::string::npos); \
    }                                                                        \
    EXPECT_FALSE(caught_exception);                                          \
  }

TEST(NanInfUtils, Functions) {
  // test all methods
  auto tensor = paddle::experimental::full(
      {3, 4}, std::numeric_limits<double>::quiet_NaN(), phi::DataType::FLOAT64);
  CHECK_NAN_INF(tensor);
  auto tensor1 = paddle::experimental::full(
      {3, 4}, std::numeric_limits<double>::quiet_NaN(), phi::DataType::FLOAT64);
  auto two_tensors = std::make_tuple(tensor, tensor1);
  CHECK_NAN_INF(two_tensors);
  auto tensor2 = paddle::experimental::full(
      {3, 4}, std::numeric_limits<double>::quiet_NaN(), phi::DataType::FLOAT64);
  auto three_tensors = std::make_tuple(tensor, tensor1, tensor2);
  CHECK_NAN_INF(three_tensors);
  auto tensor3 = paddle::experimental::full(
      {3, 4}, std::numeric_limits<double>::quiet_NaN(), phi::DataType::FLOAT64);
  auto four_tensors = std::make_tuple(tensor, tensor1, tensor2, tensor3);
  CHECK_NAN_INF(four_tensors);
  auto tensor4 = paddle::experimental::full(
      {3, 4}, std::numeric_limits<double>::quiet_NaN(), phi::DataType::FLOAT64);
  auto five_tensors =
      std::make_tuple(tensor, tensor1, tensor2, tensor3, tensor4);
  CHECK_NAN_INF(five_tensors);
  auto tensor5 = paddle::experimental::full(
      {3, 4}, std::numeric_limits<double>::quiet_NaN(), phi::DataType::FLOAT64);
  auto six_tensors =
      std::make_tuple(tensor, tensor1, tensor2, tensor3, tensor4, tensor5);
  CHECK_NAN_INF(six_tensors);
  std::vector<paddle::experimental::Tensor> tensor_vec;
  tensor_vec.emplace_back(tensor);
  tensor_vec.emplace_back(tensor1);
  CHECK_NAN_INF(tensor_vec);
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      small_vec;
  small_vec.emplace_back(tensor_vec);
  CHECK_NAN_INF(small_vec);
  // test selected_rows
  paddle::experimental::Tensor tensor_sr;
  auto sr = std::make_shared<phi::SelectedRows>();
  *sr->mutable_value() =
      *(static_cast<const phi::DenseTensor*>(tensor.impl().get()));
  tensor_sr.set_impl(sr);
  CHECK_NAN_INF(tensor_sr);
  // test other tensor
  auto tensor_str = paddle::experimental::strings::empty({3, 4});
  CHECK_NO_NAN_INF(tensor_str);
}

}  // namespace egr
