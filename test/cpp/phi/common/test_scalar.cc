// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <complex>
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "paddle/phi/common/scalar.h"

namespace phi {
namespace tests {

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

TEST(Scalar, Formatting) {
  paddle::experimental::Scalar s;

  s = paddle::experimental::Scalar(static_cast<float>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(float32(");

  s = paddle::experimental::Scalar(static_cast<double>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(float64(");

  s = paddle::experimental::Scalar(static_cast<int>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(int32(");

  s = paddle::experimental::Scalar(static_cast<int64_t>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(int64(");

  s = paddle::experimental::Scalar(static_cast<bool>(true));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(bool(");

  s = paddle::experimental::Scalar(std::complex<float>(42.1, 42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(complex64(");

  s = paddle::experimental::Scalar(std::complex<double>(42.1, 42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(complex128(");

  s = paddle::experimental::Scalar(static_cast<phi::float16>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(float16(");

  s = paddle::experimental::Scalar(static_cast<phi::bfloat16>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(bfloat16(");

  s = paddle::experimental::Scalar(static_cast<int8_t>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(int8(");

  s = paddle::experimental::Scalar(static_cast<int16_t>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(int16(");

  s = paddle::experimental::Scalar(static_cast<uint8_t>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(uint8(");

  s = paddle::experimental::Scalar(static_cast<uint16_t>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(uint16(");

  s = paddle::experimental::Scalar(static_cast<uint32_t>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(uint32(");

  s = paddle::experimental::Scalar(static_cast<uint64_t>(42.1));
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(uint64(");

  std::stringstream ss;
  s = paddle::experimental::Scalar(static_cast<uint64_t>(42.1));
  ss << s;
  ASSERT_PRED2(StartsWith, s.ToString(), "Scalar(uint64(");
}

TEST(Scalar, Equality) {
  auto s_bool = paddle::experimental::Scalar(static_cast<bool>(true));

  auto s_int8 = paddle::experimental::Scalar(static_cast<int8_t>(42.1));
  auto s_int16 = paddle::experimental::Scalar(static_cast<int16_t>(42.1));
  auto s_int32 = paddle::experimental::Scalar(static_cast<int32_t>(42.1));
  auto s_int64 = paddle::experimental::Scalar(static_cast<int64_t>(42.1));

  auto s_uint8 = paddle::experimental::Scalar(static_cast<uint8_t>(42.1));
  auto s_uint16 = paddle::experimental::Scalar(static_cast<uint16_t>(42.1));
  auto s_uint32 = paddle::experimental::Scalar(static_cast<uint32_t>(42.1));
  auto s_uint64 = paddle::experimental::Scalar(static_cast<uint64_t>(42.1));

  auto s_float16 =
      paddle::experimental::Scalar(static_cast<phi::float16>(42.1));
  auto s_bfloat16 =
      paddle::experimental::Scalar(static_cast<phi::bfloat16>(42.1));
  auto s_float = paddle::experimental::Scalar(static_cast<float>(42.1));
  auto s_double = paddle::experimental::Scalar(static_cast<double>(42.1));

  auto s_cfloat = paddle::experimental::Scalar(std::complex<float>(42.1, 42.1));
  auto s_cdouble =
      paddle::experimental::Scalar(std::complex<double>(42.1, 42.1));

  ASSERT_EQ(s_bool, s_bool);

  ASSERT_EQ(s_int8, s_int8);
  ASSERT_EQ(s_int16, s_int16);
  ASSERT_EQ(s_int32, s_int32);
  ASSERT_EQ(s_int64, s_int64);

  ASSERT_EQ(s_uint8, s_uint8);
  ASSERT_EQ(s_uint16, s_uint16);
  ASSERT_EQ(s_uint32, s_uint32);
  ASSERT_EQ(s_uint64, s_uint64);

  ASSERT_EQ(s_float16, s_float16);
  ASSERT_EQ(s_bfloat16, s_bfloat16);
  ASSERT_EQ(s_float, s_float);
  ASSERT_EQ(s_double, s_double);

  ASSERT_EQ(s_cfloat, s_cfloat);
  ASSERT_EQ(s_cdouble, s_cdouble);

  ASSERT_NE(s_float, s_double);
}

TEST(Scalar, WrapAsScalars) {
  std::vector<int32_t> v{1, 2, 3};
  auto out = paddle::experimental::WrapAsScalars(v);
  ASSERT_EQ(out[0].dtype(), phi::DataType::INT32);
  ASSERT_EQ(out[0].to<int32_t>(), 1);
  ASSERT_EQ(out[1].to<int32_t>(), 2);
  ASSERT_EQ(out[2].to<int32_t>(), 3);
}
}  // namespace tests
}  // namespace phi
