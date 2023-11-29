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

#include "gtest/gtest.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
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
  ASSERT_EQ(::phi::DataType::BFLOAT16,
            TransToPaddleDataType(::cinn::common::BF16()));
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

TEST(HelperFunction, PaddleAttributeToStringPODValue) {
  paddle::framework::Attribute attr1 = 1;
  ASSERT_EQ(PaddleAttributeToString(attr1), std::string("1"));

  paddle::framework::Attribute attr2 = 0.2f;
  ASSERT_EQ(PaddleAttributeToString(attr2), std::string("0.2"));

  paddle::framework::Attribute attr3 = true;
  ASSERT_EQ(PaddleAttributeToString(attr3), std::string("true"));

  paddle::framework::Attribute attr4 = std::string("string_attribute");
  ASSERT_EQ(PaddleAttributeToString(attr4), std::string("string_attribute"));
}

TEST(HelperFunction, PaddleAttributeToStringVectorValue) {
  paddle::framework::Attribute attr1 = std::vector<int>();
  ASSERT_EQ(PaddleAttributeToString(attr1), std::string(""));

  paddle::framework::Attribute attr2 = std::vector<int>{1, 2, 3, 4, 5};
  ASSERT_EQ(PaddleAttributeToString(attr2), std::string("[1, 2, 3, 4, 5]"));

  paddle::framework::Attribute attr3 =
      std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  ASSERT_EQ(PaddleAttributeToString(attr3),
            std::string("[0.1, 0.2, 0.3, 0.4, 0.5]"));

  paddle::framework::Attribute attr4 =
      std::vector<bool>{true, false, true, false, false};
  ASSERT_EQ(PaddleAttributeToString(attr4),
            std::string("[true, false, true, false, false]"));

  paddle::framework::Attribute attr5 =
      std::vector<std::string>{"a", "b", "c", "d", "e"};
  ASSERT_EQ(PaddleAttributeToString(attr5), std::string("[a, b, c, d, e]"));
}

}  // namespace paddle::framework::paddle2cinn
