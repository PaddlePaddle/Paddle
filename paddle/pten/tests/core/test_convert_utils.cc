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

#include "gtest/gtest.h"
#include "paddle/pten/core/convert_utils.h"

namespace pten {
namespace tests {

TEST(ConvertUtils, DataType) {
  // enum -> proto
  CHECK(pten::TransToProtoVarType(paddle::DataType::FLOAT64) ==
        paddle::framework::proto::VarType::FP64);
  CHECK(pten::TransToProtoVarType(paddle::DataType::FLOAT32) ==
        paddle::framework::proto::VarType::FP32);
  CHECK(pten::TransToProtoVarType(paddle::DataType::UINT8) ==
        paddle::framework::proto::VarType::UINT8);
  CHECK(pten::TransToProtoVarType(paddle::DataType::INT8) ==
        paddle::framework::proto::VarType::INT8);
  CHECK(pten::TransToProtoVarType(paddle::DataType::INT32) ==
        paddle::framework::proto::VarType::INT32);
  CHECK(pten::TransToProtoVarType(paddle::DataType::INT64) ==
        paddle::framework::proto::VarType::INT64);
  CHECK(pten::TransToProtoVarType(paddle::DataType::INT16) ==
        paddle::framework::proto::VarType::INT16);
  CHECK(pten::TransToProtoVarType(paddle::DataType::BOOL) ==
        paddle::framework::proto::VarType::BOOL);
  CHECK(pten::TransToProtoVarType(paddle::DataType::COMPLEX64) ==
        paddle::framework::proto::VarType::COMPLEX64);
  CHECK(pten::TransToProtoVarType(paddle::DataType::COMPLEX128) ==
        paddle::framework::proto::VarType::COMPLEX128);
  CHECK(pten::TransToProtoVarType(paddle::DataType::FLOAT16) ==
        paddle::framework::proto::VarType::FP16);
  // proto -> enum
  CHECK(pten::TransToPtenDataType(paddle::framework::proto::VarType::FP64) ==
        paddle::DataType::FLOAT64);
  CHECK(pten::TransToPtenDataType(paddle::framework::proto::VarType::FP32) ==
        paddle::DataType::FLOAT32);
  CHECK(pten::TransToPtenDataType(paddle::framework::proto::VarType::INT64) ==
        paddle::DataType::INT64);
  CHECK(pten::TransToPtenDataType(paddle::framework::proto::VarType::INT32) ==
        paddle::DataType::INT32);
  CHECK(pten::TransToPtenDataType(paddle::framework::proto::VarType::INT8) ==
        paddle::DataType::INT8);
  CHECK(pten::TransToPtenDataType(paddle::framework::proto::VarType::UINT8) ==
        paddle::DataType::UINT8);
  CHECK(pten::TransToPtenDataType(paddle::framework::proto::VarType::INT16) ==
        paddle::DataType::INT16);
  CHECK(pten::TransToPtenDataType(paddle::framework::proto::VarType::BOOL) ==
        paddle::DataType::BOOL);
  CHECK(
      pten::TransToPtenDataType(paddle::framework::proto::VarType::COMPLEX64) ==
      paddle::DataType::COMPLEX64);
  CHECK(pten::TransToPtenDataType(
            paddle::framework::proto::VarType::COMPLEX128) ==
        paddle::DataType::COMPLEX128);
  CHECK(pten::TransToPtenDataType(paddle::framework::proto::VarType::FP16) ==
        paddle::DataType::FLOAT16);
}

}  // namespace tests
}  // namespace pten
