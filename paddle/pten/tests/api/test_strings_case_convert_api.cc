/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See
the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <memory>

#include "paddle/pten/api/include/api.h"

#include "paddle/pten/api/include/strings_api.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/core/string_tensor.h"
#include "paddle/pten/kernels/strings/strings_empty_kernel.h"

namespace paddle {
namespace tests {

using pten::GPUPlace;
using pten::CPUPlace;
using pten::CPUContext;
using pten::StringTensor;
using pten::StringTensorMeta;

TEST(API, case_convert) {
  auto cpu = CPUPlace();
  const auto alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(cpu);
  // 1. create tensor
  const pten::DDim dims({1, 2});
  StringTensorMeta meta(dims);
  auto cpu_strings_x = std::make_shared<pten::StringTensor>(
      alloc.get(), pten::StringTensorMeta(meta));
  pstring* cpu_strings_x_data = cpu_strings_x->mutable_data(cpu);
  std::string strs[] = {"A Short Pstring.",
                        "A Large Pstring Whose Length Is Longer Than 22."};
  for (int i = 0; i < 2; ++i) {
    cpu_strings_x_data[i] = strs[i];
  }
  // 2. get expected results
  std::string expected_results[] = {strs[0], strs[0], strs[1], strs[1]};
  std::transform(
      strs[0].begin(), strs[0].end(), expected_results[0].begin(), ::tolower);
  std::transform(
      strs[0].begin(), strs[0].end(), expected_results[1].begin(), ::toupper);
  std::transform(
      strs[1].begin(), strs[1].end(), expected_results[2].begin(), ::tolower);
  std::transform(
      strs[1].begin(), strs[1].end(), expected_results[3].begin(), ::toupper);
  // 3. test API, ascii encoding
  paddle::experimental::Tensor x(cpu_strings_x);
  auto lower_out = paddle::experimental::strings::lower(x, "");
  auto upper_out = paddle::experimental::strings::upper(x, "");

  auto lower_tensor =
      std::dynamic_pointer_cast<pten::StringTensor>(lower_out.impl());
  auto upper_tensor =
      std::dynamic_pointer_cast<pten::StringTensor>(upper_out.impl());
  ASSERT_EQ(lower_tensor->dims(), dims);
  ASSERT_EQ(upper_tensor->dims(), dims);

  auto lower_tensor_ptr = lower_tensor->data();
  auto upper_tensor_ptr = upper_tensor->data();

  const std::string cpu_results[] = {lower_tensor_ptr[0].data(),
                                     upper_tensor_ptr[0].data(),
                                     lower_tensor_ptr[1].data(),
                                     upper_tensor_ptr[1].data()};

  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(cpu_results[i], expected_results[i]);
  }
}

TEST(API, case_convert_utf8) {
  auto cpu = CPUPlace();
  const auto alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(cpu);
  // 1. create tensor
  const pten::DDim dims({1, 2});
  StringTensorMeta meta(dims);
  auto cpu_strings_x = std::make_shared<pten::StringTensor>(
      alloc.get(), pten::StringTensorMeta(meta));
  pstring* cpu_strings_x_data = cpu_strings_x->mutable_data(cpu);
  std::string strs[] = {"óÓsscHloëË", "óÓsscHloëËóÓsscHloëËóÓsscHloëË"};
  for (int i = 0; i < 2; ++i) {
    cpu_strings_x_data[i] = strs[i];
  }
  // 2. get expected results
  std::string expected_results[] = {"óósschloëë",
                                    "ÓÓSSCHLOËË",
                                    "óósschloëëóósschloëëóósschloëë",
                                    "ÓÓSSCHLOËËÓÓSSCHLOËËÓÓSSCHLOËË"};
  // 3. test API, ascii encoding
  paddle::experimental::Tensor x(cpu_strings_x);
  auto lower_out = paddle::experimental::strings::lower(x, "utf-8");
  auto upper_out = paddle::experimental::strings::upper(x, "utf-8");

  auto lower_tensor =
      std::dynamic_pointer_cast<pten::StringTensor>(lower_out.impl());
  auto upper_tensor =
      std::dynamic_pointer_cast<pten::StringTensor>(upper_out.impl());
  ASSERT_EQ(lower_tensor->dims(), dims);
  ASSERT_EQ(upper_tensor->dims(), dims);

  auto lower_tensor_ptr = lower_tensor->data();
  auto upper_tensor_ptr = upper_tensor->data();

  const char* cpu_results[] = {lower_tensor_ptr[0].data(),
                               upper_tensor_ptr[0].data(),
                               lower_tensor_ptr[1].data(),
                               upper_tensor_ptr[1].data()};

  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(std::string(cpu_results[i]), expected_results[i]);
  }
}

}  // namespace tests
}  // namespace paddle
