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
  oss << phi::DataType::UNDEFINED;
  EXPECT_EQ(oss.str(), "Undefined");
  oss.str("");
  oss << phi::DataType::BOOL;
  EXPECT_EQ(oss.str(), "bool");
  oss.str("");
  oss << phi::DataType::INT8;
  EXPECT_EQ(oss.str(), "int8");
  oss.str("");
  oss << phi::DataType::UINT8;
  EXPECT_EQ(oss.str(), "uint8");
  oss.str("");
  oss << phi::DataType::INT16;
  EXPECT_EQ(oss.str(), "int16");
  oss.str("");
  oss << phi::DataType::INT32;
  EXPECT_EQ(oss.str(), "int32");
  oss.str("");
  oss << phi::DataType::INT64;
  EXPECT_EQ(oss.str(), "int64");
  oss.str("");
  oss << phi::DataType::BFLOAT16;
  EXPECT_EQ(oss.str(), "bfloat16");
  oss.str("");
  oss << phi::DataType::FLOAT16;
  EXPECT_EQ(oss.str(), "float16");
  oss.str("");
  oss << phi::DataType::FLOAT32;
  EXPECT_EQ(oss.str(), "float32");
  oss.str("");
  oss << phi::DataType::FLOAT64;
  EXPECT_EQ(oss.str(), "float64");
  oss.str("");
  oss << phi::DataType::COMPLEX64;
  EXPECT_EQ(oss.str(), "complex64");
  oss.str("");
  oss << phi::DataType::COMPLEX128;
  EXPECT_EQ(oss.str(), "complex128");
  oss.str("");
  oss << phi::DataType::PSTRING;
  EXPECT_EQ(oss.str(), "pstring");
  oss.str("");
  try {
    oss << phi::DataType::NUM_DATA_TYPES;
  } catch (const std::exception& exception) {
    std::string ex_msg = exception.what();
    EXPECT_TRUE(ex_msg.find("Invalid enum data type") != std::string::npos);
  }
}

TEST(TypeTraits, Complex) {
  EXPECT_EQ(phi::dtype::ToReal(phi::DataType::COMPLEX64),
            phi::DataType::FLOAT32);
  EXPECT_EQ(phi::dtype::ToReal(phi::DataType::COMPLEX128),
            phi::DataType::FLOAT64);
  EXPECT_EQ(phi::dtype::ToReal(phi::DataType::FLOAT32), phi::DataType::FLOAT32);

  EXPECT_EQ(phi::dtype::ToComplex(phi::DataType::FLOAT32),
            phi::DataType::COMPLEX64);
  EXPECT_EQ(phi::dtype::ToComplex(phi::DataType::FLOAT64),
            phi::DataType::COMPLEX128);
  EXPECT_EQ(phi::dtype::ToComplex(phi::DataType::COMPLEX64),
            phi::DataType::COMPLEX64);
}

}  // namespace tests
}  // namespace phi
