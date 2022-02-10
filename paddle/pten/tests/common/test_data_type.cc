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

#include "paddle/pten/api/ext/exception.h"
#include "paddle/pten/common/data_type.h"

namespace pten {
namespace tests {

TEST(DataType, OStream) {
  std::ostringstream oss;
  oss << pten::DataType::UNDEFINED;
  EXPECT_EQ(oss.str(), "Undefined");
  oss.str("");
  oss << pten::DataType::BOOL;
  EXPECT_EQ(oss.str(), "bool");
  oss.str("");
  oss << pten::DataType::INT8;
  EXPECT_EQ(oss.str(), "int8");
  oss.str("");
  oss << pten::DataType::UINT8;
  EXPECT_EQ(oss.str(), "uint8");
  oss.str("");
  oss << pten::DataType::INT16;
  EXPECT_EQ(oss.str(), "int16");
  oss.str("");
  oss << pten::DataType::INT32;
  EXPECT_EQ(oss.str(), "int32");
  oss.str("");
  oss << pten::DataType::INT64;
  EXPECT_EQ(oss.str(), "int64");
  oss.str("");
  oss << pten::DataType::BFLOAT16;
  EXPECT_EQ(oss.str(), "bfloat16");
  oss.str("");
  oss << pten::DataType::FLOAT16;
  EXPECT_EQ(oss.str(), "float16");
  oss.str("");
  oss << pten::DataType::FLOAT32;
  EXPECT_EQ(oss.str(), "float32");
  oss.str("");
  oss << pten::DataType::FLOAT64;
  EXPECT_EQ(oss.str(), "float64");
  oss.str("");
  oss << pten::DataType::COMPLEX64;
  EXPECT_EQ(oss.str(), "complex64");
  oss.str("");
  oss << pten::DataType::COMPLEX128;
  EXPECT_EQ(oss.str(), "complex128");
  oss.str("");
  try {
    oss << pten::DataType::NUM_DATA_TYPES;
  } catch (const std::exception& exception) {
    std::string ex_msg = exception.what();
    EXPECT_TRUE(ex_msg.find("Invalid enum data type") != std::string::npos);
  }
}

}  // namespace tests
}  // namespace pten
