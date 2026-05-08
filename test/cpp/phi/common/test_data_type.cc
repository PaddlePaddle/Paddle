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

#include <iostream>
#include <sstream>

#include "paddle/common/exception.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/type_traits.h"

namespace phi {
namespace tests {

TEST(DataType, OStream) {
  std::ostringstream oss;
  oss << DataType::UNDEFINED;
  EXPECT_EQ(oss.str(), "Undefined");
  oss.str("");
  oss << DataType::BOOL;
  EXPECT_EQ(oss.str(), "bool");
  oss.str("");
  oss << DataType::INT8;
  EXPECT_EQ(oss.str(), "int8");
  oss.str("");
  oss << DataType::UINT8;
  EXPECT_EQ(oss.str(), "uint8");
  oss.str("");
  oss << DataType::INT16;
  EXPECT_EQ(oss.str(), "int16");
  oss.str("");
  oss << DataType::INT32;
  EXPECT_EQ(oss.str(), "int32");
  oss.str("");
  oss << DataType::INT64;
  EXPECT_EQ(oss.str(), "int64");
  oss.str("");
  oss << DataType::BFLOAT16;
  EXPECT_EQ(oss.str(), "bfloat16");
  oss.str("");
  oss << DataType::FLOAT16;
  EXPECT_EQ(oss.str(), "float16");
  oss.str("");
  oss << DataType::FLOAT32;
  EXPECT_EQ(oss.str(), "float32");
  oss.str("");
  oss << DataType::FLOAT64;
  EXPECT_EQ(oss.str(), "float64");
  oss.str("");
  oss << DataType::COMPLEX64;
  EXPECT_EQ(oss.str(), "complex64");
  oss.str("");
  oss << DataType::COMPLEX128;
  EXPECT_EQ(oss.str(), "complex128");
  oss.str("");
  oss << DataType::PSTRING;
  EXPECT_EQ(oss.str(), "pstring");
  oss.str("");
  try {
    oss << DataType::NUM_DATA_TYPES;
  } catch (const std::exception& exception) {
    std::string ex_msg = exception.what();
    EXPECT_TRUE(ex_msg.find("Invalid enum data type") != std::string::npos);
  }
}

TEST(TypeTraits, Complex) {
  EXPECT_EQ(dtype::ToReal(DataType::COMPLEX64), DataType::FLOAT32);
  EXPECT_EQ(dtype::ToReal(DataType::COMPLEX128), DataType::FLOAT64);
  EXPECT_EQ(dtype::ToReal(DataType::FLOAT32), DataType::FLOAT32);

  EXPECT_EQ(dtype::ToComplex(DataType::FLOAT32), DataType::COMPLEX64);
  EXPECT_EQ(dtype::ToComplex(DataType::FLOAT64), DataType::COMPLEX128);
  EXPECT_EQ(dtype::ToComplex(DataType::COMPLEX64), DataType::COMPLEX64);
}

}  // namespace tests
}  // namespace phi
