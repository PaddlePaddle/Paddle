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

#include "paddle/phi/tests/ops/test_op_signature.h"

#include <gtest/gtest.h>
#include <memory>
#include <unordered_set>

#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/ops/compat/signatures.h"

namespace phi {
namespace tests {

// The unittests in this file are just order to pass the CI-Coverage，
// so it isn't necessary to check the all cases.

TEST(ARG_MAP, fill_constant) {
  TestArgumentMappingContext arg_case1(
      {"ShapeTensor", "ValueTensor"}, {}, {}, {}, {"Out"});
  auto signature1 =
      OpUtilsMap::Instance().GetArgumentMappingFn("fill_constant")(arg_case1);
  ASSERT_EQ(signature1.name, "full_sr");

  TestArgumentMappingContext arg_case2(
      {"ShapeTensor"},
      {},
      {{"str_value", paddle::any{std::string{"10"}}}},
      {},
      {"Out"});
  auto signature2 =
      OpUtilsMap::Instance().GetArgumentMappingFn("fill_constant")(arg_case2);
  ASSERT_EQ(signature2.name, "full_sr");

  TestArgumentMappingContext arg_case3(
      {"ShapeTensor"},
      {},
      {{"value", paddle::any{0}}, {"str_value", paddle::any{std::string{""}}}},
      {},
      {"Out"});
  auto signature3 =
      OpUtilsMap::Instance().GetArgumentMappingFn("fill_constant")(arg_case3);
  ASSERT_EQ(signature3.name, "full_sr");

  TestArgumentMappingContext arg_case4(
      {"ShapeTensorList", "ValueTensor"}, {}, {}, {}, {"Out"});
  auto signature4 =
      OpUtilsMap::Instance().GetArgumentMappingFn("fill_constant")(arg_case4);
  ASSERT_EQ(signature4.name, "full_sr");

  TestArgumentMappingContext arg_case5(
      {"ShapeTensorList"},
      {},
      {{"str_value", paddle::any{std::string{"10"}}}},
      {},
      {"Out"});
  auto signature5 =
      OpUtilsMap::Instance().GetArgumentMappingFn("fill_constant")(arg_case5);
  ASSERT_EQ(signature5.name, "full_sr");

  TestArgumentMappingContext arg_case6(
      {"ShapeTensorList"},
      {},
      {{"value", paddle::any{0}}, {"str_value", paddle::any{std::string{""}}}},
      {},
      {"Out"});
  auto signature6 =
      OpUtilsMap::Instance().GetArgumentMappingFn("fill_constant")(arg_case6);
  ASSERT_EQ(signature6.name, "full_sr");

  TestArgumentMappingContext arg_case7(
      {"ValueTensor"},
      {},
      {{"shape", paddle::any{std::vector<int64_t>{2, 3}}}},
      {},
      {"Out"});
  auto signature7 =
      OpUtilsMap::Instance().GetArgumentMappingFn("fill_constant")(arg_case7);
  ASSERT_EQ(signature7.name, "full_sr");

  TestArgumentMappingContext arg_case8(
      {},
      {},
      {{"shape", paddle::any{std::vector<int64_t>{2, 3}}},
       {"value", paddle::any{0}},
       {"str_value", paddle::any{std::string{""}}}},
      {},
      {"Out"});
  auto signature8 =
      OpUtilsMap::Instance().GetArgumentMappingFn("fill_constant")(arg_case8);
  ASSERT_EQ(signature8.name, "full_sr");

  TestArgumentMappingContext arg_case9(
      {},
      {},
      {{"shape", paddle::any{std::vector<int64_t>{2, 3}}},
       {"str_value", paddle::any{std::string{"10"}}}},
      {},
      {"Out"});
  auto signature9 =
      OpUtilsMap::Instance().GetArgumentMappingFn("fill_constant")(arg_case9);
  ASSERT_EQ(signature9.name, "full_sr");
}

}  // namespace tests
}  // namespace phi
