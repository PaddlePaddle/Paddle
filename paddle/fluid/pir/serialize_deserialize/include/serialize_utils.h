// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

#include "paddle/common/layout.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/serialize_deserialize/include/schema.h"
#include "paddle/fluid/pir/serialize_deserialize/include/third_party.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace pir {
#define COMPRESS_DIALECT_NAME(attr_template)           \
  pir::DialectIdMap::Instance()->GetCompressDialectId( \
      (attr_template).dialect().name())

class AttrTypeWriter {
 public:
  static Json WriteBuiltInAttr(const pir::Attribute& attr);

  static Json WriteBuiltInType(const pir::Type& type);

  static Json WritePaddleOperatorAttr(const pir::Attribute& attr);

  static Json WritePaddleOperatorType(const pir::Type& type);
};
/** serializeTypeToJson is a template function to serialize
 * a pir type to a json object. a pir type may have value or no value
 * Value free types only have ID, while value based types have
 * DATA in addition to ID.
 *
 * If a new pir type is added, which needs to be serialized,
 * it must have a name() method, returning a string which
 * should be different from other types' names.
 * (The name template is t_dialectname_typename).
 * Note: The prefixes t are assumed to represent 'type'.
 *
 * If the pir type has value, it should have a data() method,
 * which returns the value of type. The data() method is better
 * suited to return TYPE  which supported by json like std::vector,
 * std::string, int, float and so on. if not, serailizeTypeToJson
 * need to be specialized.
 */

template <typename T>
Json serializeTypeToJson(const T& type) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(type) + "." + type.name();
  return json_obj;
}

/** serializeAttrToJson is a template function to serialize
 * pir attribute to json object. pir attribute usually have
 * value, so it's json object has DATA and ID.
 *
 * If a new pir attr is added, which needs to be serialized,
 * it must have a name() method, returning a string which
 * should be different from other types' names.
 * (The name template is a_dialectname_typename).
 * Note: The prefixes a are assumed to represent 'attribute'.
 *
 * It also need have a data() method, which returns the value of
 * attribute. The data() method is better suited to return TYPE
 * which supported by json like std::vector, std::string, int,
 * float and so on. if not, serailizeAttrToJson
 * need to be specialized.
 */

template <typename T>
Json serializeAttrToJson(const T& attr) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name();
  json_obj[DATA] = attr.data();
  return json_obj;
}

#define SERIALIZE_ATTR_TO_JSON(type, data)                          \
  template <>                                                       \
  Json serializeAttrToJson<type>(const type& attr) {                \
    Json json_obj;                                                  \
    json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name(); \
    json_obj[DATA] = data;                                          \
    return json_obj;                                                \
  }

SERIALIZE_ATTR_TO_JSON(pir::StrAttribute, attr.AsString());

SERIALIZE_ATTR_TO_JSON(pir::Complex64Attribute,
                       std::vector({attr.data().real, attr.data().imag}));
SERIALIZE_ATTR_TO_JSON(pir::Complex128Attribute,
                       std::vector({attr.data().real, attr.data().imag}));
SERIALIZE_ATTR_TO_JSON(paddle::dialect::IntArrayAttribute,
                       attr.data().GetData());
SERIALIZE_ATTR_TO_JSON(paddle::dialect::DataTypeAttribute,
                       phi::DataTypeToString(attr.data()));
SERIALIZE_ATTR_TO_JSON(paddle::dialect::DataLayoutAttribute,
                       common::DataLayoutToString(attr.data()));
template <>
Json serializeAttrToJson<paddle::dialect::ScalarAttribute>(
    const paddle::dialect::ScalarAttribute& attr) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name();

  Json content = Json::array();
  auto scalar = attr.data();
  auto dtype_ = scalar.dtype();
  content.push_back(DataTypeToString(dtype_));

  if (dtype_ == phi::DataType::FLOAT32) {
    content.push_back(scalar.to<float>());
  } else if (dtype_ == phi::DataType::INT32) {
    content.push_back(scalar.to<int32_t>());
  } else if (dtype_ == phi::DataType::FLOAT64) {
    content.push_back(scalar.to<double>());
  } else if (dtype_ == phi::DataType::INT8) {
    content.push_back(scalar.to<int8_t>());
  } else if (dtype_ == phi::DataType::FLOAT16 ||
             dtype_ == phi::DataType::UINT16 ||
             dtype_ == phi::DataType::BFLOAT16) {
    content.push_back(scalar.to<uint16_t>());
  } else if (dtype_ == phi::DataType::INT16) {
    content.push_back(scalar.to<int16_t>());
  } else if (dtype_ == phi::DataType::INT64) {
    content.push_back(scalar.to<int64_t>());
  } else if (dtype_ == phi::DataType::UINT8) {
    content.push_back(scalar.to<uint8_t>());
  } else if (dtype_ == phi::DataType::UINT32) {
    content.push_back(scalar.to<uint32_t>());
  } else if (dtype_ == phi::DataType::UINT64) {
    content.push_back(scalar.to<uint64_t>());
  } else if (dtype_ == phi::DataType::BOOL) {
    content.push_back(scalar.to<bool>());
  } else if (dtype_ == phi::DataType::COMPLEX64) {
    content.push_back(scalar.to<phi::dtype::complex<float>>().real);
    content.push_back(scalar.to<phi::dtype::complex<float>>().imag);
  } else if (dtype_ == phi::DataType::COMPLEX128) {
    content.push_back(scalar.to<phi::dtype::complex<double>>().real);
    content.push_back(scalar.to<phi::dtype::complex<double>>().imag);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Invalid tensor data type `", dtype_, "`."));
  }
  json_obj[DATA] = content;
  return json_obj;
}

template <>
Json serializeAttrToJson<paddle::dialect::PlaceAttribute>(
    const paddle::dialect::PlaceAttribute& attr) {
  Json json_obj;
  json_obj[ID] = COMPRESS_DIALECT_NAME(attr) + "." + attr.name();
  Json content = Json::array();
  auto place = attr.data();
  content.push_back(static_cast<int8_t>(place.GetType()));
  content.push_back(place.GetDeviceId());    // int8_t
  content.push_back(place.GetDeviceType());  // string
  json_obj[DATA] = content;
  return json_obj;
}

Json writeType(const pir::Type& type) {
  Json type_json = Json::object();
  if (!type) {
    type_json[ID] = NULL_TYPE;
    return type_json;
  }
  if (type.dialect().name() == pir::BuiltinDialect::name()) {
    VLOG(6) << "write BuiltinType ... ";
    return AttrTypeWriter::WriteBuiltInType(type);
  } else if (type.dialect().name() ==
             paddle::dialect::OperatorDialect::name()) {
    VLOG(6) << "write PaddleOperatorType ... ";
    return AttrTypeWriter::WritePaddleOperatorType(type);
  } else {
    PADDLE_ENFORCE(
        false, phi::errors::InvalidArgument("Unknown Type %s when write type"));
  }
  VLOG(8) << "Finish write Type ... ";

  return type_json;
}

SERIALIZE_ATTR_TO_JSON(pir::TypeAttribute, writeType(attr.data()));

Json writeAttr(const pir::Attribute& attr) {
  if (attr.dialect().name() == pir::BuiltinDialect::name()) {
    VLOG(8) << "write BuiltinAttr ... ";
    return AttrTypeWriter::WriteBuiltInAttr(attr);
  } else if (attr.dialect().name() ==
             paddle::dialect::OperatorDialect::name()) {
    VLOG(8) << "write PaddleOperatorAttr ... ";
    return AttrTypeWriter::WritePaddleOperatorAttr(attr);
  } else {
    PADDLE_ENFORCE(
        false, phi::errors::InvalidArgument("Unknown Attr %s when write attr"));
  }

  VLOG(8) << "Finish write attr ... ";

  return Json::object();
}

Json AttrTypeWriter::WriteBuiltInAttr(const pir::Attribute& attr) {
  Json attr_json = Json::object();
  if (attr.isa<pir::BoolAttribute>()) {
    VLOG(8) << "write BoolAttribute .";
    return pir::serializeAttrToJson<pir::BoolAttribute>(
        attr.dyn_cast<pir::BoolAttribute>());
  } else if (attr.isa<pir::FloatAttribute>()) {
    VLOG(8) << "write FloatAttribute .";
    return pir::serializeAttrToJson<pir::FloatAttribute>(
        attr.dyn_cast<pir::FloatAttribute>());
  } else if (attr.isa<pir::DoubleAttribute>()) {
    VLOG(8) << "write DoubleAttribute .";
    return pir::serializeAttrToJson<pir::DoubleAttribute>(
        attr.dyn_cast<pir::DoubleAttribute>());
  } else if (attr.isa<pir::Int32Attribute>()) {
    VLOG(8) << "write Int32Attribute .";
    return pir::serializeAttrToJson<pir::Int32Attribute>(
        attr.dyn_cast<pir::Int32Attribute>());
  } else if (attr.isa<pir::Int64Attribute>()) {
    VLOG(8) << "write Int64Attribute .";
    return pir::serializeAttrToJson<pir::Int64Attribute>(
        attr.dyn_cast<pir::Int64Attribute>());
  } else if (attr.isa<pir::IndexAttribute>()) {
    VLOG(8) << "write IndexAttribute .";
    return pir::serializeAttrToJson<pir::IndexAttribute>(
        attr.dyn_cast<pir::IndexAttribute>());
  } else if (attr.isa<pir::ArrayAttribute>()) {
    VLOG(8) << "write ArrayAttribute .";
    auto attr_ = attr.dyn_cast<pir::ArrayAttribute>();
    Json val = Json::array();
    for (size_t i = 0; i < attr_.size(); i++) {
      val.push_back(writeAttr(attr_.at(i)));
    }
    attr_json[ID] = COMPRESS_DIALECT_NAME(attr_) + "." + attr_.name();
    attr_json[DATA] = val;
    return attr_json;
  } else if (attr.isa<pir::TypeAttribute>()) {
    VLOG(8) << "write TypeAttribute .";
    return pir::serializeAttrToJson<pir::TypeAttribute>(
        attr.dyn_cast<pir::TypeAttribute>());
  } else if (attr.isa<pir::TensorNameAttribute>()) {
    VLOG(8) << "write TensorNameAttribute .";
    return pir::serializeAttrToJson<pir::TensorNameAttribute>(
        attr.dyn_cast<pir::TensorNameAttribute>());
  } else if (attr.isa<pir::Complex64Attribute>()) {
    VLOG(8) << "write Complex64Attribute .";
    return pir::serializeAttrToJson<pir::Complex64Attribute>(
        attr.dyn_cast<pir::Complex64Attribute>());
  } else if (attr.isa<pir::Complex128Attribute>()) {
    VLOG(8) << "write Complex128Attribute .";
    return pir::serializeAttrToJson<pir::Complex128Attribute>(
        attr.dyn_cast<pir::Complex128Attribute>());
  } else if (attr.isa<pir::StrAttribute>()) {
    VLOG(8) << "write StrAttribute .";
    return pir::serializeAttrToJson<pir::StrAttribute>(
        attr.dyn_cast<pir::StrAttribute>());
  } else {
    PADDLE_ENFORCE(false,
                   phi::errors::InvalidArgument(
                       "Unknown Attr %s when write Buitin dialect attr"));
  }
  return attr_json;
}

Json AttrTypeWriter::WriteBuiltInType(const pir::Type& type) {
  Json type_json = Json::object();
  if (type.isa<pir::BoolType>()) {
    VLOG(8) << "Write BoolType ... ";
    return pir::serializeTypeToJson<pir::BoolType>(
        type.dyn_cast<pir::BoolType>());
  } else if (type.isa<pir::BFloat16Type>()) {
    VLOG(8) << "Write BFloat16Type ... ";
    return pir::serializeTypeToJson<pir::BFloat16Type>(
        type.dyn_cast<pir::BFloat16Type>());
  } else if (type.isa<pir::Float16Type>()) {
    VLOG(8) << "Write Float16Type ... ";
    return pir::serializeTypeToJson<pir::Float16Type>(
        type.dyn_cast<pir::Float16Type>());
  } else if (type.isa<pir::Float32Type>()) {
    VLOG(8) << "Write Float32Type ... ";
    return pir::serializeTypeToJson<pir::Float32Type>(
        type.dyn_cast<pir::Float32Type>());
  } else if (type.isa<pir::Float64Type>()) {
    VLOG(8) << "Write Float64Type ... ";
    return pir::serializeTypeToJson<pir::Float64Type>(
        type.dyn_cast<pir::Float64Type>());
  } else if (type.isa<pir::Int8Type>()) {
    VLOG(8) << "Write Int8Type ... ";
    return pir::serializeTypeToJson<pir::Int8Type>(
        type.dyn_cast<pir::Int8Type>());
  } else if (type.isa<pir::UInt8Type>()) {
    VLOG(8) << "Write UInt8Type ... ";
    return pir::serializeTypeToJson<pir::UInt8Type>(
        type.dyn_cast<pir::UInt8Type>());
  } else if (type.isa<pir::Int16Type>()) {
    VLOG(8) << "Write Int16Type ... ";
    return pir::serializeTypeToJson<pir::Int16Type>(
        type.dyn_cast<pir::Int16Type>());
  } else if (type.isa<pir::Int32Type>()) {
    VLOG(8) << "Write Int32Type ... ";
    return pir::serializeTypeToJson<pir::Int32Type>(
        type.dyn_cast<pir::Int32Type>());
  } else if (type.isa<pir::Int64Type>()) {
    VLOG(8) << "Write Int64Type ... ";
    return pir::serializeTypeToJson<pir::Int64Type>(
        type.dyn_cast<pir::Int64Type>());
  } else if (type.isa<pir::IndexType>()) {
    VLOG(8) << "Write IndexType ... ";
    return pir::serializeTypeToJson<pir::IndexType>(
        type.dyn_cast<pir::IndexType>());
  } else if (type.isa<pir::Complex64Type>()) {
    VLOG(8) << "Write Complex64Type ... ";
    return pir::serializeTypeToJson<pir::Complex64Type>(
        type.dyn_cast<pir::Complex64Type>());
  } else if (type.isa<pir::Complex128Type>()) {
    VLOG(8) << "Write Complex128Type ... ";
    return pir::serializeTypeToJson<pir::Complex128Type>(
        type.dyn_cast<pir::Complex128Type>());
    // NOTE(Ruting) those Types need call writeType which make build error
    //  when use template func serializeTypeToJson
  } else if (type.isa<pir::VectorType>()) {
    VLOG(8) << "Write VectorType ... ";
    auto type_ = type.dyn_cast<pir::VectorType>();
    type_json[ID] = COMPRESS_DIALECT_NAME(type_) + "." + type_.name();
    Json content = Json::array();
    for (auto type_x : type_.data()) {
      content.push_back(writeType(type_x));
    }
    type_json[DATA] = content;
    return type_json;
  } else if (type.isa<pir::DenseTensorType>()) {
    VLOG(8) << "Write DenseTensorType ... ";
    auto type_ = type.dyn_cast<pir::DenseTensorType>();

    type_json[ID] = COMPRESS_DIALECT_NAME(type_) + "." + type_.name();
    Json content = Json::array();
    content.push_back(writeType(type_.dtype()));

    std::vector<int64_t> dims_;
    for (auto i = 0; i < type_.dims().size(); i++) {
      dims_.push_back(type_.dims().at(i));
    }
    content.push_back(dims_);

    content.push_back(DataLayoutToString(type_.data_layout()));

    content.push_back(type_.lod());

    content.push_back(type_.offset());
    type_json[DATA] = content;
    return type_json;
  } else {
    PADDLE_ENFORCE(false,
                   phi::errors::InvalidArgument(
                       "Unknown Type when write builtin dialect type"));
  }
  return type_json;
}

Json AttrTypeWriter::WritePaddleOperatorAttr(const pir::Attribute& attr) {
  if (attr.isa<paddle::dialect::IntArrayAttribute>()) {
    VLOG(8) << "write IntArrayAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::IntArrayAttribute>(
        attr.dyn_cast<paddle::dialect::IntArrayAttribute>());
  } else if (attr.isa<paddle::dialect::ScalarAttribute>()) {
    VLOG(8) << "write ScalarAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::ScalarAttribute>(
        attr.dyn_cast<paddle::dialect::ScalarAttribute>());
  } else if (attr.isa<paddle::dialect::DataTypeAttribute>()) {
    VLOG(8) << "write DataTypeAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::DataTypeAttribute>(
        attr.dyn_cast<paddle::dialect::DataTypeAttribute>());
  } else if (attr.isa<paddle::dialect::PlaceAttribute>()) {
    VLOG(8) << "write PlaceAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::PlaceAttribute>(
        attr.dyn_cast<paddle::dialect::PlaceAttribute>());
  } else if (attr.isa<paddle::dialect::DataLayoutAttribute>()) {
    VLOG(8) << "write DataLayoutAttribute .";
    return pir::serializeAttrToJson<paddle::dialect::DataLayoutAttribute>(
        attr.dyn_cast<paddle::dialect::DataLayoutAttribute>());
  } else {
    PADDLE_ENFORCE(
        false,
        phi::errors::InvalidArgument(
            "Unknown Attr %s when write paddle.operatordialect attr"));
  }
  return Json::object();
}

Json AttrTypeWriter::WritePaddleOperatorType(const pir::Type& type) {
  PADDLE_ENFORCE(false,
                 phi::errors::InvalidArgument(
                     "Unknown Type when write paddle.operatordialect type"));

  return Json::object();
}

}  // namespace pir
