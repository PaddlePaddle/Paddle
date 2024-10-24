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

#include "paddle/fluid/framework/convert_utils.h"
#include "gtest/gtest.h"
#include "paddle/common/enforce.h"

namespace phi {
namespace tests {

TEST(ConvertUtils, DataType) {
  // enum -> proto
  PADDLE_ENFORCE_EQ(
      paddle::framework::TransToProtoVarType(paddle::DataType::FLOAT64),
      paddle::framework::proto::VarType::FP64,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::DataType::FLOAT64 to "
          "paddle::framework::proto::VarType::FP64"));
  PADDLE_ENFORCE_EQ(
      paddle::framework::TransToProtoVarType(paddle::DataType::FLOAT32),
      paddle::framework::proto::VarType::FP32,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::DataType::FLOAT32 to "
          "paddle::framework::proto::VarType::FP32"));
  PADDLE_ENFORCE_EQ(
      paddle::framework::TransToProtoVarType(paddle::DataType::UINT8),
      paddle::framework::proto::VarType::UINT8,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::DataType::UINT8 to "
          "paddle::framework::proto::VarType::UINT8"));
  PADDLE_ENFORCE_EQ(
      paddle::framework::TransToProtoVarType(paddle::DataType::INT8),
      paddle::framework::proto::VarType::INT8,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::DataType::INT8 to "
          "paddle::framework::proto::VarType::INT8"));
  PADDLE_ENFORCE_EQ(
      paddle::framework::TransToProtoVarType(paddle::DataType::INT32),
      paddle::framework::proto::VarType::INT32,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::DataType::INT32 to "
          "paddle::framework::proto::VarType::INT32"));
  PADDLE_ENFORCE_EQ(
      paddle::framework::TransToProtoVarType(paddle::DataType::INT64),
      paddle::framework::proto::VarType::INT64,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::DataType::INT64 to "
          "paddle::framework::proto::VarType::INT64"));
  PADDLE_ENFORCE_EQ(
      paddle::framework::TransToProtoVarType(paddle::DataType::INT16),
      paddle::framework::proto::VarType::INT16,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::DataType::INT16 to "
          "paddle::framework::proto::VarType::INT16"));
  PADDLE_ENFORCE_EQ(
      paddle::framework::TransToProtoVarType(paddle::DataType::BOOL),
      paddle::framework::proto::VarType::BOOL,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::DataType::BOOL to "
          "paddle::framework::proto::VarType::BOOL"));
  PADDLE_ENFORCE_EQ(
      paddle::framework::TransToProtoVarType(paddle::DataType::COMPLEX64),
      paddle::framework::proto::VarType::COMPLEX64,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::DataType::COMPLEX64 to "
          "paddle::framework::proto::VarType::COMPLEX64"));
  PADDLE_ENFORCE_EQ(
      paddle::framework::TransToProtoVarType(paddle::DataType::COMPLEX128),
      paddle::framework::proto::VarType::COMPLEX128,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::DataType::COMPLEX128 to "
          "paddle::framework::proto::VarType::COMPLEX128"));
  PADDLE_ENFORCE_EQ(
      paddle::framework::TransToProtoVarType(paddle::DataType::FLOAT16),
      paddle::framework::proto::VarType::FP16,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::DataType::FLOAT16 to "
          "paddle::framework::proto::VarType::FP16"));

  // proto -> enum
  PADDLE_ENFORCE_EQ(
      phi::TransToPhiDataType(paddle::framework::proto::VarType::FP64),
      paddle::DataType::FLOAT64,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::framework::proto::VarType::FP64 to "
          "paddle::DataType::FLOAT64"));
  PADDLE_ENFORCE_EQ(
      phi::TransToPhiDataType(paddle::framework::proto::VarType::FP32),
      paddle::DataType::FLOAT32,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::framework::proto::VarType::FP32 to "
          "paddle::DataType::FLOAT32"));
  PADDLE_ENFORCE_EQ(
      phi::TransToPhiDataType(paddle::framework::proto::VarType::INT64),
      paddle::DataType::INT64,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::framework::proto::VarType::INT64 to "
          "paddle::DataType::INT64"));
  PADDLE_ENFORCE_EQ(
      phi::TransToPhiDataType(paddle::framework::proto::VarType::INT32),
      paddle::DataType::INT32,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::framework::proto::VarType::INT32 to "
          "paddle::DataType::INT32"));
  PADDLE_ENFORCE_EQ(
      phi::TransToPhiDataType(paddle::framework::proto::VarType::INT8),
      paddle::DataType::INT8,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::framework::proto::VarType::INT8 to "
          "paddle::DataType::INT8"));
  PADDLE_ENFORCE_EQ(
      phi::TransToPhiDataType(paddle::framework::proto::VarType::UINT8),
      paddle::DataType::UINT8,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::framework::proto::VarType::UINT8 to "
          "paddle::DataType::UINT8"));
  PADDLE_ENFORCE_EQ(
      phi::TransToPhiDataType(paddle::framework::proto::VarType::INT16),
      paddle::DataType::INT16,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::framework::proto::VarType::INT16 to "
          "paddle::DataType::INT16"));
  PADDLE_ENFORCE_EQ(
      phi::TransToPhiDataType(paddle::framework::proto::VarType::BOOL),
      paddle::DataType::BOOL,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::framework::proto::VarType::BOOL to "
          "paddle::DataType::BOOL"));
  PADDLE_ENFORCE_EQ(
      phi::TransToPhiDataType(paddle::framework::proto::VarType::COMPLEX64),
      paddle::DataType::COMPLEX64,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::framework::proto::VarType::COMPLEX64 to "
          "paddle::DataType::COMPLEX64"));
  PADDLE_ENFORCE_EQ(
      phi::TransToPhiDataType(paddle::framework::proto::VarType::COMPLEX128),
      paddle::DataType::COMPLEX128,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::framework::proto::VarType::COMPLEX128 to "
          "paddle::DataType::COMPLEX128"));
  PADDLE_ENFORCE_EQ(
      phi::TransToPhiDataType(paddle::framework::proto::VarType::FP16),
      paddle::DataType::FLOAT16,
      ::common::errors::InvalidArgument(
          "Failed to convert paddle::framework::proto::VarType::FP16 to "
          "paddle::DataType::FLOAT16"));
}

}  // namespace tests
}  // namespace phi
