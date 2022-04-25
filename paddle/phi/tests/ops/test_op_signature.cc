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
  auto signature1 = (*OpUtilsMap::Instance().GetArgumentMappingFn(
      "fill_constant"))(arg_case1);
  ASSERT_EQ(signature1.name, "full_sr");

  TestArgumentMappingContext arg_case2(
      {"ShapeTensor"},
      {},
      {{"str_value", paddle::any{std::string{"10"}}}},
      {},
      {"Out"});
  auto signature2 = (*OpUtilsMap::Instance().GetArgumentMappingFn(
      "fill_constant"))(arg_case2);
  ASSERT_EQ(signature2.name, "full_sr");

  TestArgumentMappingContext arg_case3(
      {"ShapeTensor"},
      {},
      {{"value", paddle::any{0}}, {"str_value", paddle::any{std::string{""}}}},
      {},
      {"Out"});
  auto signature3 = (*OpUtilsMap::Instance().GetArgumentMappingFn(
      "fill_constant"))(arg_case3);
  ASSERT_EQ(signature3.name, "full_sr");

  TestArgumentMappingContext arg_case4(
      {"ShapeTensorList", "ValueTensor"}, {}, {}, {}, {"Out"});
  auto signature4 = (*OpUtilsMap::Instance().GetArgumentMappingFn(
      "fill_constant"))(arg_case4);
  ASSERT_EQ(signature4.name, "full_sr");

  TestArgumentMappingContext arg_case5(
      {"ShapeTensorList"},
      {},
      {{"str_value", paddle::any{std::string{"10"}}}},
      {},
      {"Out"});
  auto signature5 = (*OpUtilsMap::Instance().GetArgumentMappingFn(
      "fill_constant"))(arg_case5);
  ASSERT_EQ(signature5.name, "full_sr");

  TestArgumentMappingContext arg_case6(
      {"ShapeTensorList"},
      {},
      {{"value", paddle::any{0}}, {"str_value", paddle::any{std::string{""}}}},
      {},
      {"Out"});
  auto signature6 = (*OpUtilsMap::Instance().GetArgumentMappingFn(
      "fill_constant"))(arg_case6);
  ASSERT_EQ(signature6.name, "full_sr");

  TestArgumentMappingContext arg_case7(
      {"ValueTensor"},
      {},
      {{"shape", paddle::any{std::vector<int64_t>{2, 3}}}},
      {},
      {"Out"});
  auto signature7 = (*OpUtilsMap::Instance().GetArgumentMappingFn(
      "fill_constant"))(arg_case7);
  ASSERT_EQ(signature7.name, "full_sr");

  TestArgumentMappingContext arg_case8(
      {},
      {},
      {{"shape", paddle::any{std::vector<int64_t>{2, 3}}},
       {"value", paddle::any{0}},
       {"str_value", paddle::any{std::string{""}}}},
      {},
      {"Out"});
  auto signature8 = (*OpUtilsMap::Instance().GetArgumentMappingFn(
      "fill_constant"))(arg_case8);
  ASSERT_EQ(signature8.name, "full_sr");

  TestArgumentMappingContext arg_case9(
      {},
      {},
      {{"shape", paddle::any{std::vector<int64_t>{2, 3}}},
       {"str_value", paddle::any{std::string{"10"}}}},
      {},
      {"Out"});
  auto signature9 = (*OpUtilsMap::Instance().GetArgumentMappingFn(
      "fill_constant"))(arg_case9);
  ASSERT_EQ(signature9.name, "full_sr");
}

TEST(ARG_MAP, set_value) {
  TestArgumentMappingContext arg_case(
      {"Input", "StartsTensorList", "EndsTensorList", "StepsTensorList"},
      {},
      {{"fp32_values", paddle::any{std::vector<float>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case1(
      {"Input", "StartsTensorList", "EndsTensorList", "StepsTensorList"},
      {},
      {{"fp64_values", paddle::any{std::vector<double>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case1)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case2(
      {"Input", "StartsTensorList", "EndsTensorList", "StepsTensorList"},
      {},
      {{"int32_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case2)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case3(
      {"Input", "StartsTensorList", "EndsTensorList", "StepsTensorList"},
      {},
      {{"int64_values", paddle::any{std::vector<int64_t>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case3)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case4(
      {"Input", "StartsTensorList", "EndsTensorList", "StepsTensorList"},
      {},
      {{"bool_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case4)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case5(
      {"Input", "StartsTensorList", "EndsTensorList", "ValueTensor"},
      {},
      {},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case5)
          .name,
      "set_value_with_tensor");

  TestArgumentMappingContext arg_case6(
      {"Input", "StartsTensorList", "EndsTensorList"},
      {},
      {{"fp64_values", paddle::any{std::vector<double>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case6)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case7(
      {"Input", "StartsTensorList", "EndsTensorList"},
      {},
      {{"int32_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case7)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case8(
      {"Input", "StartsTensorList", "EndsTensorList"},
      {},
      {{"int64_values", paddle::any{std::vector<int64_t>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case8)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case9(
      {"Input", "StartsTensorList", "EndsTensorList"},
      {},
      {{"bool_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case9)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case10(
      {"Input", "StartsTensorList", "StepsTensorList", "ValueTensor"},
      {},
      {},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case10)
          .name,
      "set_value_with_tensor");

  TestArgumentMappingContext arg_case11(
      {"Input", "StartsTensorList", "StepsTensorList"},
      {},
      {{"fp64_values", paddle::any{std::vector<double>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case11)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case12(
      {"Input", "StartsTensorList", "StepsTensorList"},
      {},
      {{"int32_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case12)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case13(
      {"Input", "StartsTensorList", "StepsTensorList"},
      {},
      {{"int64_values", paddle::any{std::vector<int64_t>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case13)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case14(
      {"Input", "StartsTensorList", "StepsTensorList"},
      {},
      {{"bool_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case14)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case15(
      {"Input", "StartsTensorList", "ValueTensor"}, {}, {}, {"Out"}, {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case15)
          .name,
      "set_value_with_tensor");

  TestArgumentMappingContext arg_case16(
      {"Input", "StartsTensorList", "StepsTensorList"},
      {},
      {{"fp32_values", paddle::any{std::vector<float>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case16)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case17(
      {"Input", "StartsTensorList", "StepsTensorList"},
      {},
      {{"fp64_values", paddle::any{std::vector<double>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case17)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case18(
      {"Input", "StartsTensorList", "StepsTensorList"},
      {},
      {{"int32_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case18)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case19(
      {"Input", "StartsTensorList", "StepsTensorList"},
      {},
      {{"int64_values", paddle::any{std::vector<int64_t>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case19)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case20(
      {"Input", "StartsTensorList", "StepsTensorList"},
      {},
      {{"bool_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case20)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case21(
      {"Input", "EndsTensorList", "StepsTensorList", "ValueTensor"},
      {},
      {},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case21)
          .name,
      "set_value_with_tensor");

  TestArgumentMappingContext arg_case22(
      {"Input", "EndsTensorList", "StepsTensorList"},
      {},
      {{"fp64_values", paddle::any{std::vector<double>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case22)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case23(
      {"Input", "EndsTensorList", "StepsTensorList"},
      {},
      {{"int32_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case23)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case24(
      {"Input", "EndsTensorList", "StepsTensorList"},
      {},
      {{"int64_values", paddle::any{std::vector<int64_t>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case24)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case25(
      {"Input", "EndsTensorList", "StepsTensorList"},
      {},
      {{"bool_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case25)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case26(
      {"Input", "EndsTensorList", "ValueTensor"}, {}, {}, {"Out"}, {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case26)
          .name,
      "set_value_with_tensor");

  TestArgumentMappingContext arg_case27(
      {"Input", "EndsTensorList"},
      {},
      {{"fp32_values", paddle::any{std::vector<float>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case27)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case28(
      {"Input", "EndsTensorList"},
      {},
      {{"fp64_values", paddle::any{std::vector<double>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case28)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case29(
      {"Input", "EndsTensorList"},
      {},
      {{"int32_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case29)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case30(
      {"Input", "EndsTensorList"},
      {},
      {{"int64_values", paddle::any{std::vector<int64_t>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case30)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case31(
      {"Input", "EndsTensorList"},
      {},
      {{"bool_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case31)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case32(
      {"Input", "StepsTensorList", "ValueTensor"}, {}, {}, {"Out"}, {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case32)
          .name,
      "set_value_with_tensor");

  TestArgumentMappingContext arg_case33(
      {"Input", "StepsTensorList"},
      {},
      {{"fp32_values", paddle::any{std::vector<float>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case33)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case34(
      {"Input", "StepsTensorList"},
      {},
      {{"fp64_values", paddle::any{std::vector<double>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case34)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case35(
      {"Input", "StepsTensorList"},
      {},
      {{"int32_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case35)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case36(
      {"Input", "StepsTensorList"},
      {},
      {{"int64_values", paddle::any{std::vector<int64_t>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case36)
          .name,
      "set_value");

  TestArgumentMappingContext arg_case37(
      {"Input", "StepsTensorList"},
      {},
      {{"bool_values", paddle::any{std::vector<int>{1}}}},
      {"Out"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value"))(arg_case37)
          .name,
      "set_value");
}

TEST(ARG_MAP, set_value_grad) {
  TestArgumentMappingContext arg_case(
      {"Out@GRAD", "StartsTensorList", "EndsTensorList"},
      {},
      {},
      {"Input@GRAD", "ValueTensor@GRAD"},
      {});
  ASSERT_EQ(
      (*OpUtilsMap::Instance().GetArgumentMappingFn("set_value_grad"))(arg_case)
          .name,
      "set_value_grad");

  TestArgumentMappingContext arg_case1(
      {"Out@GRAD", "StartsTensorList", "StepsTensorList"},
      {},
      {},
      {"Input@GRAD", "ValueTensor@GRAD"},
      {});
  ASSERT_EQ((*OpUtilsMap::Instance().GetArgumentMappingFn("set_value_grad"))(
                arg_case1)
                .name,
            "set_value_grad");

  TestArgumentMappingContext arg_case2({"Out@GRAD", "StartsTensorList"},
                                       {},
                                       {},
                                       {"Input@GRAD", "ValueTensor@GRAD"},
                                       {});
  ASSERT_EQ((*OpUtilsMap::Instance().GetArgumentMappingFn("set_value_grad"))(
                arg_case2)
                .name,
            "set_value_grad");

  TestArgumentMappingContext arg_case3(
      {"Out@GRAD", "EndsTensorList", "StepsTensorList"},
      {},
      {},
      {"Input@GRAD", "ValueTensor@GRAD"},
      {});
  ASSERT_EQ((*OpUtilsMap::Instance().GetArgumentMappingFn("set_value_grad"))(
                arg_case3)
                .name,
            "set_value_grad");

  TestArgumentMappingContext arg_case4({"Out@GRAD", "EndsTensorList"},
                                       {},
                                       {},
                                       {"Input@GRAD", "ValueTensor@GRAD"},
                                       {});
  ASSERT_EQ((*OpUtilsMap::Instance().GetArgumentMappingFn("set_value_grad"))(
                arg_case4)
                .name,
            "set_value_grad");

  TestArgumentMappingContext arg_case5({"Out@GRAD", "StepsTensorList"},
                                       {},
                                       {},
                                       {"Input@GRAD", "ValueTensor@GRAD"},
                                       {});
  ASSERT_EQ((*OpUtilsMap::Instance().GetArgumentMappingFn("set_value_grad"))(
                arg_case5)
                .name,
            "set_value_grad");
}

TEST(ARG_MAP, allclose) {
  TestArgumentMappingContext arg_case1(
      {"Input", "Other", "Rtol"},
      {},
      {{"atol", paddle::any(std::string{"1e-8"})},
       {"equal_nan", paddle::any(false)}},
      {"Out"},
      {});
  auto signature1 =
      (*OpUtilsMap::Instance().GetArgumentMappingFn("allclose"))(arg_case1);
  ASSERT_EQ(signature1.name, "allclose");
  ASSERT_EQ(signature1.attr_names[0], "Rtol");

  TestArgumentMappingContext arg_case2(
      {"Input", "Other", "Atol"},
      {},
      {{"rtol", paddle::any(std::string{"1e-5"})},
       {"equal_nan", paddle::any(false)}},
      {"Out"},
      {});
  auto signature2 =
      (*OpUtilsMap::Instance().GetArgumentMappingFn("allclose"))(arg_case2);
  ASSERT_EQ(signature2.name, "allclose");
  ASSERT_EQ(signature2.attr_names[1], "Atol");
}

TEST(ARG_MAP, reshape) {
  TestArgumentMappingContext arg_case1({"X", "ShapeTensor"}, {}, {}, {"Out"});
  auto signature1 =
      (*OpUtilsMap::Instance().GetArgumentMappingFn("reshape2"))(arg_case1);
  ASSERT_EQ(signature1.name, "reshape");

  TestArgumentMappingContext arg_case2({"X", "Shape"}, {}, {}, {"Out"});
  auto signature2 =
      (*OpUtilsMap::Instance().GetArgumentMappingFn("reshape2"))(arg_case2);
  ASSERT_EQ(signature2.name, "reshape");

  TestArgumentMappingContext arg_case3(
      {"X"}, {}, {{"shape", paddle::any(std::vector<int>({1, 2}))}}, {"Out"});
  auto signature3 =
      (*OpUtilsMap::Instance().GetArgumentMappingFn("reshape2"))(arg_case3);
  ASSERT_EQ(signature3.name, "reshape");
}

}  // namespace tests
}  // namespace phi
