//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/attribute.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/any.h"

TEST(Attribute, GetAttrValueToAny) {
  paddle::framework::Attribute x_int(100);
  auto rlt_int = paddle::framework::GetAttrValue(x_int);
  EXPECT_EQ(paddle::any_cast<int>(rlt_int), 100);

  float float_value = 3.14;
  paddle::framework::Attribute x_float(float_value);
  auto rlt_float = paddle::framework::GetAttrValue(x_float);
  EXPECT_NEAR(paddle::any_cast<float>(rlt_float), 3.14, 1e-6);

  std::string str_value("test");
  paddle::framework::Attribute x_str(str_value);
  auto rlt_str = paddle::framework::GetAttrValue(x_str);
  EXPECT_EQ(paddle::any_cast<std::string>(rlt_str), "test");

  std::vector<int> vec_int_var(2, 100);
  paddle::framework::Attribute x_vec_int = vec_int_var;
  auto rlt_vec_int = paddle::framework::GetAttrValue(x_vec_int);
  auto vec_int = paddle::any_cast<std::vector<int>>(rlt_vec_int);
  EXPECT_EQ(vec_int.size(), 2UL);
  EXPECT_EQ(vec_int[0], 100);
  EXPECT_EQ(vec_int[1], 100);

  std::vector<float> vec_float_var(2, 3.14);
  paddle::framework::Attribute x_vec_float = vec_float_var;
  auto rlt_vec_float = paddle::framework::GetAttrValue(x_vec_float);
  auto vec_float = paddle::any_cast<std::vector<float>>(rlt_vec_float);
  EXPECT_EQ(vec_float.size(), 2UL);
  EXPECT_NEAR(vec_float[0], 3.14, 1e-6);
  EXPECT_NEAR(vec_float[1], 3.14, 1e-6);

  std::vector<std::string> vec_str_var(2, "test");
  paddle::framework::Attribute x_vec_str = vec_str_var;
  auto rlt_vec_str = paddle::framework::GetAttrValue(x_vec_str);
  auto vec_str = paddle::any_cast<std::vector<std::string>>(rlt_vec_str);
  EXPECT_EQ(vec_str.size(), 2UL);
  EXPECT_EQ(vec_str[0], "test");
  EXPECT_EQ(vec_str[1], "test");

  paddle::framework::Attribute x_bool(true);
  auto rlt_bool = paddle::framework::GetAttrValue(x_bool);
  EXPECT_EQ(paddle::any_cast<bool>(rlt_bool), true);

  std::vector<bool> vec_bool_var(2, true);
  paddle::framework::Attribute x_vec_bool = vec_bool_var;
  auto rlt_vec_bool = paddle::framework::GetAttrValue(x_vec_bool);
  auto vec_bool = paddle::any_cast<std::vector<bool>>(rlt_vec_bool);
  EXPECT_EQ(vec_bool.size(), 2UL);
  EXPECT_EQ(vec_bool[0], true);
  EXPECT_EQ(vec_bool[1], true);

  paddle::framework::VarDesc var_desc("axis");
  paddle::framework::Attribute var_attr(&var_desc);
  auto rlt_var_attr = paddle::framework::GetAttrValue(var_attr);
  auto var_desc_ptr =
      paddle::any_cast<paddle::framework::VarDesc *>(rlt_var_attr);
  EXPECT_NE(var_desc_ptr, nullptr);
  EXPECT_EQ(var_desc_ptr->Name(), var_desc.Name());

  paddle::framework::VarDesc var2_desc("prob");
  std::vector<paddle::framework::VarDesc *> vars_desc{&var_desc, &var2_desc};
  paddle::framework::Attribute vars_attr(vars_desc);

  auto rlt_vars_attr = paddle::framework::GetAttrValue(vars_attr);
  auto rlt_vars_desc =
      paddle::any_cast<std::vector<paddle::framework::VarDesc *>>(
          rlt_vars_attr);
  EXPECT_EQ(rlt_vars_desc.size(), vars_desc.size());
  EXPECT_EQ(rlt_vars_desc[0]->Name(), vars_desc[0]->Name());
  EXPECT_EQ(rlt_vars_desc[1]->Name(), vars_desc[1]->Name());

  paddle::framework::ProgramDesc prog;
  paddle::framework::proto::BlockDesc proto_block;
  paddle::framework::BlockDesc block_desc(&prog, &proto_block);
  paddle::framework::Attribute x_block_desc(&block_desc);
  auto rlt_block_desc = paddle::framework::GetAttrValue(x_block_desc);
  auto block_desc_ptr =
      paddle::any_cast<paddle::framework::BlockDesc *>(rlt_block_desc);
  EXPECT_NE(block_desc_ptr, nullptr);

  std::vector<paddle::framework::BlockDesc *> vec_block_desc_var;
  vec_block_desc_var.emplace_back(&block_desc);
  paddle::framework::Attribute x_vec_block_desc(vec_block_desc_var);
  auto rlt_vec_block_desc = paddle::framework::GetAttrValue(x_vec_block_desc);
  auto vec_block_desc =
      paddle::any_cast<std::vector<paddle::framework::BlockDesc *>>(
          rlt_vec_block_desc);
  EXPECT_EQ(vec_block_desc.size(), 1UL);
  EXPECT_NE(vec_block_desc[0], nullptr);

  int64_t int64_value = 100;
  paddle::framework::Attribute x_int64(int64_value);
  auto rlt_int64 = paddle::framework::GetAttrValue(x_int64);
  EXPECT_EQ(paddle::any_cast<int64_t>(rlt_int64), 100);

  std::vector<int64_t> vec_int64_var(2, 100);
  paddle::framework::Attribute x_vec_int64 = vec_int64_var;
  auto rlt_vec_int64 = paddle::framework::GetAttrValue(x_vec_int64);
  auto vec_int64 = paddle::any_cast<std::vector<int64_t>>(rlt_vec_int64);
  EXPECT_EQ(vec_int64.size(), 2UL);
  EXPECT_EQ(vec_int64[0], 100);
  EXPECT_EQ(vec_int64[1], 100);

  std::vector<double> vec_double_var(2, 3.14);
  paddle::framework::Attribute x_vec_double = vec_double_var;
  auto rlt_vec_double = paddle::framework::GetAttrValue(x_vec_double);
  auto vec_double = paddle::any_cast<std::vector<double>>(rlt_vec_double);
  EXPECT_EQ(vec_double.size(), 2UL);
  EXPECT_NEAR(vec_double[0], 3.14, 1e-6);
  EXPECT_NEAR(vec_double[1], 3.14, 1e-6);

  double x_double_val = 42.1;
  paddle::framework::Attribute x_double(x_double_val);
  ASSERT_EQ(AttrTypeID(x_double), paddle::framework::proto::FLOAT64);
  EXPECT_NEAR(
      paddle::any_cast<double>(paddle::framework::GetAttrValue(x_double)),
      42.1,
      1e-6);

  paddle::framework::Attribute x_scalar = paddle::experimental::Scalar(42.1);
  ASSERT_EQ(AttrTypeID(x_scalar), paddle::framework::proto::SCALAR);
  EXPECT_EQ(paddle::any_cast<paddle::experimental::Scalar>(
                paddle::framework::GetAttrValue(x_scalar)),
            paddle::experimental::Scalar(42.1));

  std::vector<paddle::experimental::Scalar> scalars =
      paddle::experimental::WrapAsScalars(std::vector<int64_t>{1, 2, 3});
  paddle::framework::Attribute x_scalars(scalars);
  ASSERT_EQ(AttrTypeID(x_scalars), paddle::framework::proto::SCALARS);
  auto x_extracted =
      paddle::any_cast<std::vector<paddle::experimental::Scalar>>(
          paddle::framework::GetAttrValue(x_scalars));
  EXPECT_EQ(x_extracted.size(), 3UL);
  EXPECT_EQ(x_extracted.at(0), scalars.at(0));
  EXPECT_EQ(x_extracted.at(1), scalars.at(1));
  EXPECT_EQ(x_extracted.at(2), scalars.at(2));
}

TEST(Attribute, ProtoAttrToAttribute_double) {
  paddle::framework::proto::OpDesc::Attr proto_attr_double;
  proto_attr_double.set_name("anon");
  proto_attr_double.set_type(paddle::framework::proto::FLOAT64);
  proto_attr_double.set_float64(42.1);
  paddle::framework::Attribute attr_double =
      paddle::framework::GetAttrValue(proto_attr_double);
  ASSERT_EQ(AttrTypeID(attr_double), paddle::framework::proto::FLOAT64);
}

TEST(Attribute, ProtoAttrToAttribute_scalar) {
  paddle::framework::proto::OpDesc::Attr proto_attr_scalar;
  proto_attr_scalar.set_name("anon");
  proto_attr_scalar.set_type(paddle::framework::proto::SCALAR);

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

  auto proto_scalar_bool = new paddle::framework::proto::Scalar;
  *proto_scalar_bool = paddle::framework::MakeScalarProto(s_bool);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_bool);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_int8 = new paddle::framework::proto::Scalar;
  *proto_scalar_int8 = paddle::framework::MakeScalarProto(s_int8);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_int8);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_int16 = new paddle::framework::proto::Scalar;
  *proto_scalar_int16 = paddle::framework::MakeScalarProto(s_int16);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_int16);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_int32 = new paddle::framework::proto::Scalar;
  *proto_scalar_int32 = paddle::framework::MakeScalarProto(s_int32);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_int32);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_int64 = new paddle::framework::proto::Scalar;
  *proto_scalar_int64 = paddle::framework::MakeScalarProto(s_int64);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_int64);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_uint8 = new paddle::framework::proto::Scalar;
  *proto_scalar_uint8 = paddle::framework::MakeScalarProto(s_uint8);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_uint8);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_uint16 = new paddle::framework::proto::Scalar;
  *proto_scalar_uint16 = paddle::framework::MakeScalarProto(s_uint16);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_uint16);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_uint32 = new paddle::framework::proto::Scalar;
  *proto_scalar_uint32 = paddle::framework::MakeScalarProto(s_uint32);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_uint32);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_uint64 = new paddle::framework::proto::Scalar;
  *proto_scalar_uint64 = paddle::framework::MakeScalarProto(s_uint64);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_uint64);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_float16 = new paddle::framework::proto::Scalar;
  *proto_scalar_float16 = paddle::framework::MakeScalarProto(s_float16);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_float16);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_bfloat16 = new paddle::framework::proto::Scalar;
  *proto_scalar_bfloat16 = paddle::framework::MakeScalarProto(s_bfloat16);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_bfloat16);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_float = new paddle::framework::proto::Scalar;
  *proto_scalar_float = paddle::framework::MakeScalarProto(s_float);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_float);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_double = new paddle::framework::proto::Scalar;
  *proto_scalar_double = paddle::framework::MakeScalarProto(s_double);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_double);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_cfloat = new paddle::framework::proto::Scalar;
  *proto_scalar_cfloat = paddle::framework::MakeScalarProto(s_cfloat);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_cfloat);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);

  auto proto_scalar_cdouble = new paddle::framework::proto::Scalar;
  *proto_scalar_cdouble = paddle::framework::MakeScalarProto(s_cdouble);
  proto_attr_scalar.set_allocated_scalar(proto_scalar_cdouble);
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalar)),
            paddle::framework::proto::SCALAR);
}

TEST(Attribute, ProtoAttrToAttribute_scalars) {
  paddle::framework::proto::OpDesc::Attr proto_attr_scalars;
  proto_attr_scalars.set_name("anon");
  proto_attr_scalars.set_type(paddle::framework::proto::SCALARS);

  std::vector<paddle::experimental::Scalar> scalars;
  scalars.reserve(10);
  for (int i = 0; i < 10; i++) {
    scalars.emplace_back(i);
  }
  std::vector<paddle::framework::proto::Scalar> proto_scalars;
  proto_scalars.reserve(scalars.size());
  for (const auto &item : scalars) {
    proto_scalars.emplace_back(paddle::framework::MakeScalarProto(item));
  }
  paddle::framework::VectorToRepeated(proto_scalars,
                                      proto_attr_scalars.mutable_scalars());
  ASSERT_EQ(AttrTypeID(paddle::framework::GetAttrValue(proto_attr_scalars)),
            paddle::framework::proto::SCALARS);
}

TEST(Attribute, MakeScalarFromAttribute) {
  using paddle::framework::MakeScalarFromAttribute;
  auto s_bool = true;
  auto s_int32 = static_cast<int32_t>(42.1);
  auto s_int64 = static_cast<int64_t>(42.1);

  auto s_float = static_cast<float>(42.1);
  auto s_double = static_cast<double>(42.1);

  auto s_scalar = paddle::experimental::Scalar(42.1);

  ASSERT_EQ(MakeScalarFromAttribute(paddle::framework::Attribute(s_bool)),
            paddle::experimental::Scalar(s_bool));
  ASSERT_EQ(MakeScalarFromAttribute(paddle::framework::Attribute(s_int32)),
            paddle::experimental::Scalar(s_int32));
  ASSERT_EQ(MakeScalarFromAttribute(paddle::framework::Attribute(s_int64)),
            paddle::experimental::Scalar(s_int64));
  ASSERT_EQ(MakeScalarFromAttribute(paddle::framework::Attribute(s_float)),
            paddle::experimental::Scalar(s_float));
  ASSERT_EQ(MakeScalarFromAttribute(paddle::framework::Attribute(s_double)),
            paddle::experimental::Scalar(s_double));
  ASSERT_EQ(MakeScalarFromAttribute(paddle::framework::Attribute(s_scalar)),
            s_scalar);
}

TEST(Attribute, MakeScalarsFromAttribute) {
  using paddle::framework::MakeScalarsFromAttribute;
  std::vector<bool> v_bool(4, true);
  std::vector<int> v_int(4, 42);
  std::vector<int64_t> v_int64(4, 42);
  std::vector<float> v_float(4, 42.1);
  std::vector<double> v_double(4, 42.1);
  std::vector<paddle::experimental::Scalar> v_scalar(
      4, paddle::experimental::Scalar(std::complex<float>(42.1, 42.1)));

  ASSERT_EQ(MakeScalarsFromAttribute(paddle::framework::Attribute(v_bool))[0],
            paddle::experimental::Scalar(v_bool[0]));

  ASSERT_EQ(MakeScalarsFromAttribute(paddle::framework::Attribute(v_int))[0],
            paddle::experimental::Scalar(v_int[0]));
  ASSERT_EQ(MakeScalarsFromAttribute(paddle::framework::Attribute(v_int64))[0],
            paddle::experimental::Scalar(v_int64[0]));

  ASSERT_EQ(MakeScalarsFromAttribute(paddle::framework::Attribute(v_float))[0],
            paddle::experimental::Scalar(v_float[0]));
  ASSERT_EQ(MakeScalarsFromAttribute(paddle::framework::Attribute(v_double))[0],
            paddle::experimental::Scalar(v_double[0]));
  ASSERT_EQ(MakeScalarsFromAttribute(paddle::framework::Attribute(v_scalar))[0],
            v_scalar[0]);
}
