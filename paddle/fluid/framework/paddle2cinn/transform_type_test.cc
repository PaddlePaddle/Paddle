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

#include "paddle/fluid/framework/paddle2cinn/transform_type.h"
#include "cinn/common/type.h"
#include "cinn/runtime/cinn_runtime.h"
#include "gtest/gtest.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::paddle2cinn {

TEST(TransToPaddleDataType, common_type) {
  ASSERT_EQ(::phi::DataType::BOOL,
            TransToPaddleDataType(::cinn::common::Bool()));
  ASSERT_EQ(::phi::DataType::INT8, TransToPaddleDataType(::cinn::common::I8()));
  ASSERT_EQ(::phi::DataType::INT16,
            TransToPaddleDataType(::cinn::common::I16()));
  ASSERT_EQ(::phi::DataType::INT32,
            TransToPaddleDataType(::cinn::common::I32()));
  ASSERT_EQ(::phi::DataType::INT64,
            TransToPaddleDataType(::cinn::common::I64()));
  ASSERT_EQ(::phi::DataType::UINT8,
            TransToPaddleDataType(::cinn::common::UI8()));
  ASSERT_EQ(::phi::DataType::UINT16,
            TransToPaddleDataType(::cinn::common::UI16()));
  ASSERT_EQ(::phi::DataType::UINT32,
            TransToPaddleDataType(::cinn::common::UI32()));
  ASSERT_EQ(::phi::DataType::UINT64,
            TransToPaddleDataType(::cinn::common::UI64()));
  ASSERT_EQ(::phi::DataType::FLOAT16,
            TransToPaddleDataType(::cinn::common::F16()));
  ASSERT_EQ(::phi::DataType::FLOAT32,
            TransToPaddleDataType(::cinn::common::F32()));
  ASSERT_EQ(::phi::DataType::FLOAT64,
            TransToPaddleDataType(::cinn::common::F64()));
  ASSERT_THROW(TransToPaddleDataType(::cinn::common::Type()),
               paddle::platform::EnforceNotMet);
}

TEST(TransToPaddleDataType, runtime_type) {
  ASSERT_EQ(::phi::DataType::BOOL, TransToPaddleDataType(cinn_bool_t()));
  ASSERT_EQ(::phi::DataType::INT8, TransToPaddleDataType(cinn_int8_t()));
  ASSERT_EQ(::phi::DataType::INT32, TransToPaddleDataType(cinn_int32_t()));
  ASSERT_EQ(::phi::DataType::INT64, TransToPaddleDataType(cinn_int64_t()));
  ASSERT_EQ(::phi::DataType::UINT32, TransToPaddleDataType(cinn_uint32_t()));
  ASSERT_EQ(::phi::DataType::UINT64, TransToPaddleDataType(cinn_uint64_t()));
  ASSERT_EQ(::phi::DataType::FLOAT32, TransToPaddleDataType(cinn_float32_t()));
  ASSERT_EQ(::phi::DataType::FLOAT64, TransToPaddleDataType(cinn_float64_t()));
  ASSERT_THROW(TransToPaddleDataType(cinn_type_t()),
               paddle::platform::EnforceNotMet);
}

}  // namespace paddle::framework::paddle2cinn
