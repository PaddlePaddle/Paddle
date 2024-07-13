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
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/serialize_deserialize/include/schema.h"
#include "paddle/fluid/pir/serialize_deserialize/include/third_party.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace pir {
#define DECOMPRESS_DIALECT_ID(name) \
  pir::DialectIdMap::Instance()->GetDecompressDialectId(name)

class AttrTypeReader {
 public:
  static pir::Attribute ReadBuiltInAttr(const std::string attr_name,
                                        Json* attr_json,
                                        pir::IrContext* ctx);

  static pir::Type ReadBuiltInType(const std::string type_name,
                                   Json* type_json,
                                   pir::IrContext* ctx);

  static pir::Attribute ReadPaddleOperatorAttr(const std::string attr_name,
                                               Json* attr_json,
                                               pir::IrContext* ctx);

  static pir::Type ReadPaddleOperatorType(const std::string type_name,
                                          Json* type_json,
                                          pir::IrContext* ctx);
};

template <typename T>
T deserializeTypeFromJson(Json* type_json, pir::IrContext* ctx) {
  return T::get(ctx);
}

template <typename T, typename CPP_T>
T deserializeAttrFromJson(Json* attr_json, pir::IrContext* ctx) {
  CPP_T data = attr_json->at(DATA).template get<CPP_T>();
  return T::get(ctx, data);
}

template <>
pir::Complex64Attribute deserializeAttrFromJson<pir::Complex64Attribute, float>(
    Json* attr_json, pir::IrContext* ctx) {
  Json data_json = attr_json->at(DATA);
  phi::dtype::complex<float> data =
      phi::dtype::complex(data_json.at(0).template get<float>(),
                          data_json.at(1).template get<float>());
  return pir::Complex64Attribute::get(ctx, data);
}

template <>
pir::Complex128Attribute
deserializeAttrFromJson<pir::Complex128Attribute, double>(Json* attr_json,
                                                          pir::IrContext* ctx) {
  Json data_json = attr_json->at(DATA);
  phi::dtype::complex<double> data =
      phi::dtype::complex(data_json.at(0).template get<double>(),
                          data_json.at(1).template get<double>());
  return pir::Complex128Attribute::get(ctx, data);
}

template <>
paddle::dialect::IntArrayAttribute
deserializeAttrFromJson<paddle::dialect::IntArrayAttribute,
                        std::vector<int64_t>>(Json* attr_json,
                                              pir::IrContext* ctx) {
  std::vector<int64_t> data = attr_json->at(DATA).get<std::vector<int64_t>>();
  phi::IntArray int_array = phi::IntArray(data);
  return paddle::dialect::IntArrayAttribute::get(ctx, int_array);
}

pir::Attribute deserializeAttrFromJson_scalarAttr(Json* attr_json,
                                                  pir::IrContext* ctx) {
  Json content = attr_json->at(DATA);
  phi::DataType dtype_ =
      phi::StringToDataType(content.at(0).template get<std::string>());
  phi::Scalar scalar;

  if (dtype_ == phi::DataType::FLOAT32) {
    scalar = phi::Scalar(content.at(1).template get<float>());
  } else if (dtype_ == phi::DataType::INT32) {
    scalar = phi::Scalar(content.at(1).template get<int32_t>());
  } else if (dtype_ == phi::DataType::FLOAT64) {
    scalar = phi::Scalar(content.at(1).template get<double>());
  } else if (dtype_ == phi::DataType::INT8) {
    scalar = phi::Scalar(content.at(1).template get<int8_t>());
  } else if (dtype_ == phi::DataType::FLOAT16 ||
             dtype_ == phi::DataType::UINT16 ||
             dtype_ == phi::DataType::BFLOAT16) {
    scalar = phi::Scalar(content.at(1).template get<uint16_t>());
  } else if (dtype_ == phi::DataType::INT16) {
    scalar = phi::Scalar(content.at(1).template get<int16_t>());
  } else if (dtype_ == phi::DataType::INT64) {
    scalar = phi::Scalar(content.at(1).template get<int64_t>());
  } else if (dtype_ == phi::DataType::UINT8) {
    scalar = phi::Scalar(content.at(1).template get<uint8_t>());
  } else if (dtype_ == phi::DataType::UINT32) {
    scalar = phi::Scalar(content.at(1).template get<uint32_t>());
  } else if (dtype_ == phi::DataType::UINT64) {
    scalar = phi::Scalar(content.at(1).template get<uint64_t>());
  } else if (dtype_ == phi::DataType::BOOL) {
    scalar = phi::Scalar(content.at(1).template get<bool>());
  } else if (dtype_ == phi::DataType::COMPLEX64) {
    float scalar_real = content.at(1).template get<float>();
    float scalar_imag = content.at(2).template get<float>();
    phi::dtype::complex<float> data =
        phi::dtype::complex(scalar_real, scalar_imag);
    scalar = phi::Scalar(data);
  } else if (dtype_ == phi::DataType::COMPLEX128) {
    double scalar_real = content.at(1).template get<double>();
    double scalar_imag = content.at(1).template get<double>();
    phi::dtype::complex<double> data =
        phi::dtype::complex(scalar_real, scalar_imag);
    scalar = phi::Scalar(data);
  } else {
    PADDLE_ENFORCE(false,
                   phi::errors::InvalidArgument(
                       "Invalid tensor data type `", dtype_, "`."));
  }

  return paddle::dialect::ScalarAttribute::get(ctx, scalar);
}

template <>
paddle::dialect::DataTypeAttribute
deserializeAttrFromJson<paddle::dialect::DataTypeAttribute, std::string>(
    Json* attr_json, pir::IrContext* ctx) {
  std::string data = attr_json->at(DATA).template get<std::string>();
  phi::DataType data_type = phi::StringToDataType(data);
  return paddle::dialect::DataTypeAttribute::get(ctx, data_type);
}

template <>
paddle::dialect::PlaceAttribute
deserializeAttrFromJson<paddle::dialect::PlaceAttribute, int8_t>(
    Json* attr_json, pir::IrContext* ctx) {
  Json data_json = attr_json->at(DATA);
  int8_t type_id = data_json.at(0).template get<int8_t>();
  phi::AllocationType type = static_cast<phi::AllocationType>(type_id);
  int8_t id = data_json.at(1).template get<int8_t>();                  // int8_t
  std::string dev_type = data_json.at(2).template get<std::string>();  // string
  phi::Place place = phi::Place(type, id, dev_type);
  return paddle::dialect::PlaceAttribute::get(ctx, place);
}

pir::Type parseType(Json* type_json) {
  auto type_name = type_json->at(ID).template get<std::string>();

  if (type_name == NULL_TYPE) {
    return pir::Type();
  }

  pir::IrContext* ctx = pir::IrContext::Instance();
  std::pair<std::string, std::string> name = getContentSplitByDot(type_name);

  if (DECOMPRESS_DIALECT_ID(name.first) == pir::BuiltinDialect::name()) {
    return AttrTypeReader::ReadBuiltInType(name.second, type_json, ctx);
  } else if (DECOMPRESS_DIALECT_ID(name.first) ==
             paddle::dialect::OperatorDialect::name()) {
    return AttrTypeReader::ReadPaddleOperatorType(name.second, type_json, ctx);
  } else {
    PADDLE_ENFORCE(
        false,
        phi::errors::InvalidArgument(
            "Unknown Attr %s for parse builtin dialect attr", type_name));
  }

  VLOG(8) << "Finish Parse Type ... ";

  return pir::Type();
}

template <>
pir::TypeAttribute deserializeAttrFromJson<pir::TypeAttribute, pir::Type>(
    Json* attr_json, pir::IrContext* ctx) {
  pir::Type type = parseType(&(attr_json->at(DATA)));
  return pir::TypeAttribute::get(ctx, type);
}

pir::Attribute parseAttr(Json* attr_json) {
  std::string attr_name = attr_json->at(ID).template get<std::string>();
  pir::IrContext* ctx = pir::IrContext::Instance();
  std::pair<std::string, std::string> name = getContentSplitByDot(attr_name);

  if (DECOMPRESS_DIALECT_ID(name.first) == pir::BuiltinDialect::name()) {
    return AttrTypeReader::ReadBuiltInAttr(name.second, attr_json, ctx);
  } else if (DECOMPRESS_DIALECT_ID(name.first) ==
             paddle::dialect::OperatorDialect::name()) {
    return AttrTypeReader::ReadPaddleOperatorAttr(name.second, attr_json, ctx);
  } else {
    PADDLE_ENFORCE(
        false,
        phi::errors::InvalidArgument(
            "Unknown Attr %s for parse builtin dialect attr", attr_name));
  }

  VLOG(8) << "Finish Parse Attr ... ";

  return pir::Attribute();
}

pir::Attribute AttrTypeReader::ReadBuiltInAttr(const std::string attr_name,
                                               Json* attr_json,
                                               pir::IrContext* ctx) {
  if (attr_name == pir::BoolAttribute::name()) {
    VLOG(8) << "Parse BoolAttribute .";
    return pir::deserializeAttrFromJson<pir::BoolAttribute, bool>(attr_json,
                                                                  ctx);
  } else if (attr_name == pir::FloatAttribute::name()) {
    VLOG(8) << "Parse FloatAttribute .";
    return pir::deserializeAttrFromJson<pir::FloatAttribute, float>(attr_json,
                                                                    ctx);
  } else if (attr_name == pir::DoubleAttribute::name()) {
    VLOG(8) << "Parse DoubleAttribute .";
    return pir::deserializeAttrFromJson<pir::DoubleAttribute, double>(attr_json,
                                                                      ctx);
  } else if (attr_name == pir::Int32Attribute::name()) {
    VLOG(8) << "Parse Int32Attribute .";
    return pir::deserializeAttrFromJson<pir::Int32Attribute, int32_t>(attr_json,
                                                                      ctx);
  } else if (attr_name == pir::Int64Attribute::name()) {
    VLOG(8) << "Parse Int64Attribute .";
    return pir::deserializeAttrFromJson<pir::Int64Attribute, int64_t>(attr_json,
                                                                      ctx);
  } else if (attr_name == pir::IndexAttribute::name()) {
    VLOG(8) << "Parse IndexAttribute .";
    return pir::deserializeAttrFromJson<pir::IndexAttribute, int64_t>(attr_json,
                                                                      ctx);
  } else if (attr_name == pir::ArrayAttribute::name()) {
    VLOG(8) << "Parse ArrayAttribute .";
    std::vector<pir::Attribute> val;
    for (auto& attr_ : attr_json->at(DATA)) {
      val.push_back(parseAttr(&(attr_)));
    }
    return pir::ArrayAttribute::get(ctx, val);
  } else if (attr_name == pir::TypeAttribute::name()) {
    VLOG(8) << "Parse TypeAttribute .";
    return pir::deserializeAttrFromJson<pir::TypeAttribute, pir::Type>(
        attr_json, ctx);
  } else if (attr_name == pir::TensorNameAttribute::name()) {
    VLOG(8) << "Parse TensorNameAttribute .";
    return pir::deserializeAttrFromJson<pir::TensorNameAttribute, std::string>(
        attr_json, ctx);
  } else if (attr_name == pir::Complex64Attribute::name()) {
    VLOG(8) << "Parse Complex64Attribute .";
    return pir::deserializeAttrFromJson<pir::Complex64Attribute, float>(
        attr_json, ctx);
  } else if (attr_name == pir::Complex128Attribute::name()) {
    VLOG(8) << "Parse Complex128Attribute .";
    return pir::deserializeAttrFromJson<pir::Complex128Attribute, double>(
        attr_json, ctx);
  } else if (attr_name == pir::StrAttribute::name()) {
    VLOG(8) << "Parse StrAttribute .";
    return pir::deserializeAttrFromJson<pir::StrAttribute, std::string>(
        attr_json, ctx);
  } else {
    PADDLE_ENFORCE(
        false,
        phi::errors::InvalidArgument(
            "Unknown Attr %s for parse builtin dialect attr", attr_name));
  }
  return pir::Attribute();
}

pir::Attribute AttrTypeReader::ReadPaddleOperatorAttr(
    const std::string attr_name, Json* attr_json, pir::IrContext* ctx) {
  if (attr_name == paddle::dialect::IntArrayAttribute::name()) {
    VLOG(8) << "Parse IntArrayAttribute .";
    return pir::deserializeAttrFromJson<paddle::dialect::IntArrayAttribute,
                                        std::vector<int64_t>>(attr_json, ctx);
  } else if (attr_name == paddle::dialect::ScalarAttribute::name()) {
    VLOG(8) << "Parse ScalarAttribute .";
    // this func's return type is pir::Attribute which is diffrent
    // from paddle::dialect::ScalarAttribute
    return pir::deserializeAttrFromJson_scalarAttr(attr_json, ctx);
  } else if (attr_name == paddle::dialect::DataTypeAttribute::name()) {
    VLOG(8) << "Parse DataTypeAttribute .";
    return pir::deserializeAttrFromJson<paddle::dialect::DataTypeAttribute,
                                        std::string>(attr_json, ctx);
  } else if (attr_name == paddle::dialect::PlaceAttribute::name()) {
    VLOG(8) << "Parse PlaceAttribute .";
    return pir::deserializeAttrFromJson<paddle::dialect::PlaceAttribute,
                                        int8_t>(attr_json, ctx);
  } else {
    PADDLE_ENFORCE(false,
                   phi::errors::InvalidArgument(
                       "Unknown Attr %s for parse paddleoperator dialect attr",
                       attr_name));
  }
  return pir::Attribute();
}

pir::Type AttrTypeReader::ReadBuiltInType(const std::string type_name,
                                          Json* type_json,
                                          pir::IrContext* ctx) {
  if (type_name == pir::BoolType::name()) {
    VLOG(8) << "Parse BoolType ... ";
    return pir::deserializeTypeFromJson<pir::BoolType>(type_json, ctx);
  } else if (type_name == pir::BFloat16Type::name()) {
    VLOG(8) << "Parse BFloat16Type ... ";
    return pir::deserializeTypeFromJson<pir::BFloat16Type>(type_json, ctx);
  } else if (type_name == pir::Float16Type::name()) {
    VLOG(8) << "Parse Float16Type ... ";
    return pir::deserializeTypeFromJson<pir::Float16Type>(type_json, ctx);
  } else if (type_name == pir::Float32Type::name()) {
    VLOG(8) << "Parse Float32Type ... ";
    return pir::deserializeTypeFromJson<pir::Float32Type>(type_json, ctx);
  } else if (type_name == pir::Float64Type::name()) {
    VLOG(8) << "Parse Float64Type ... ";
    return pir::deserializeTypeFromJson<pir::Float64Type>(type_json, ctx);
  } else if (type_name == pir::Int8Type::name()) {
    VLOG(8) << "Parse Int8Type ... ";
    return pir::deserializeTypeFromJson<pir::Int8Type>(type_json, ctx);
  } else if (type_name == pir::UInt8Type::name()) {
    VLOG(8) << "Parse UInt8Type ... ";
    return pir::deserializeTypeFromJson<pir::UInt8Type>(type_json, ctx);
  } else if (type_name == pir::Int16Type::name()) {
    VLOG(8) << "Parse Int16Type ... ";
    return pir::deserializeTypeFromJson<pir::Int16Type>(type_json, ctx);
  } else if (type_name == pir::Int32Type::name()) {
    VLOG(8) << "Parse Int32Type ... ";
    return pir::deserializeTypeFromJson<pir::Int32Type>(type_json, ctx);
  } else if (type_name == pir::Int64Type::name()) {
    VLOG(8) << "Parse Int64Type ... ";
    return pir::deserializeTypeFromJson<pir::Int64Type>(type_json, ctx);
  } else if (type_name == pir::IndexType::name()) {
    VLOG(8) << "Parse IndexType ... ";
    return pir::deserializeTypeFromJson<pir::IndexType>(type_json, ctx);
  } else if (type_name == pir::Complex64Type::name()) {
    VLOG(8) << "Parse Complex64Type ... ";
    return pir::deserializeTypeFromJson<pir::Complex64Type>(type_json, ctx);
  } else if (type_name == pir::Complex128Type::name()) {
    VLOG(8) << "Parse Complex128Type ... ";
    return pir::deserializeTypeFromJson<pir::Complex128Type>(type_json, ctx);
  } else if (type_name == pir::VectorType::name()) {
    VLOG(8) << "Parse VectorType ... ";
    std::vector<pir::Type> content;
    for (auto& type_x : type_json->at(DATA)) {
      content.push_back(parseType(&type_x));
    }
    return pir::VectorType::get(ctx, content);
  } else if (type_name == pir::DenseTensorType::name()) {
    VLOG(8) << "Parse DenseTensorType ... ";
    Json data_json = type_json->at(DATA);
    pir::Type dtype = parseType(&(data_json.at(0)));

    std::vector<int64_t> dims =
        data_json.at(1).template get<std::vector<int64_t>>();
    phi::DDim ddim = phi::make_ddim(dims);
    pir::DataLayout data_layout =
        common::StringToDataLayout(data_json.at(2).template get<std::string>());

    std::vector<std::vector<size_t>> lod =
        data_json.at(3).template get<std::vector<std::vector<size_t>>>();

    size_t offset = data_json.at(4).get<size_t>();
    return pir::DenseTensorType::get(
        ctx, dtype, ddim, data_layout, lod, offset);
  } else {
    PADDLE_ENFORCE(false,
                   phi::errors::InvalidArgument(
                       "Unknown Type %s for parse builtintype", type_name));
  }
  return pir::Type();
}

pir::Type AttrTypeReader::ReadPaddleOperatorType(const std::string type_name,
                                                 Json* type_json,
                                                 pir::IrContext* ctx) {
  PADDLE_ENFORCE(
      false,
      phi::errors::InvalidArgument(
          "Unknown Type %s for parse paddleoperator dialect type", type_name));
  return pir::Type();
}

}  // namespace pir
