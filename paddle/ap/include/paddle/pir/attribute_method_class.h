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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/paddle/phi/scalar_helper.h"
#include "paddle/ap/include/paddle/pir/attr_adt_type_id.h"
#include "paddle/ap/include/paddle/pir/attribute.h"

namespace ap::paddle {

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetPirAttributeClass();

template <typename T>
struct MakePirAttributeImpl;

struct MakePirAttributeImplBoolAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::BoolAttribute>
    : public MakePirAttributeImplBoolAttribute {};

struct MakePirAttributeImplComplex64Attribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::Complex64Attribute>
    : public MakePirAttributeImplComplex64Attribute {};

struct MakePirAttributeImplComplex128Attribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::Complex128Attribute>
    : public MakePirAttributeImplComplex128Attribute {};

struct MakePirAttributeImplFloatAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::FloatAttribute>
    : public MakePirAttributeImplFloatAttribute {};

struct MakePirAttributeImplDoubleAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::DoubleAttribute>
    : public MakePirAttributeImplDoubleAttribute {};

struct MakePirAttributeImplInt32Attribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::Int32Attribute>
    : public MakePirAttributeImplInt32Attribute {};

struct MakePirAttributeImplIndexAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::IndexAttribute>
    : public MakePirAttributeImplIndexAttribute {};

struct MakePirAttributeImplInt64Attribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::Int64Attribute>
    : public MakePirAttributeImplInt64Attribute {};

struct MakePirAttributeImplPointerAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::PointerAttribute>
    : public MakePirAttributeImplPointerAttribute {};

struct MakePirAttributeImplTypeAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::TypeAttribute>
    : public MakePirAttributeImplTypeAttribute {};

struct MakePirAttributeImplStrAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::StrAttribute>
    : public MakePirAttributeImplStrAttribute {};

struct MakePirAttributeImplArrayAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::ArrayAttribute>
    : public MakePirAttributeImplArrayAttribute {};

struct MakePirAttributeImplTensorNameAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::TensorNameAttribute>
    : public MakePirAttributeImplTensorNameAttribute {};

struct MakePirAttributeImplSymbolAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<pir::shape::SymbolAttribute>
    : public MakePirAttributeImplSymbolAttribute {};

struct MakePirAttributeImplKernelAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<::paddle::dialect::KernelAttribute>
    : public MakePirAttributeImplKernelAttribute {};

struct MakePirAttributeImplIntArrayAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<::paddle::dialect::IntArrayAttribute>
    : public MakePirAttributeImplIntArrayAttribute {};

struct MakePirAttributeImplScalarAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<::paddle::dialect::ScalarAttribute>
    : public MakePirAttributeImplScalarAttribute {};

struct MakePirAttributeImplDataTypeAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<::paddle::dialect::DataTypeAttribute>
    : public MakePirAttributeImplDataTypeAttribute {};

struct MakePirAttributeImplPlaceAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<::paddle::dialect::PlaceAttribute>
    : public MakePirAttributeImplPlaceAttribute {};

struct MakePirAttributeImplDataLayoutAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<::paddle::dialect::DataLayoutAttribute>
    : public MakePirAttributeImplDataLayoutAttribute {};

struct MakePirAttributeImplGroupInfoAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<::cinn::dialect::GroupInfoAttribute>
    : public MakePirAttributeImplGroupInfoAttribute {};

struct MakePirAttributeImplCINNKernelInfoAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<::cinn::dialect::CINNKernelInfoAttribute>
    : public MakePirAttributeImplCINNKernelInfoAttribute {};

struct MakePirAttributeImplUnclassifiedAttribute {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirAttributeImpl<UnclassifiedAttribute>
    : public MakePirAttributeImplUnclassifiedAttribute {};

}  // namespace ap::paddle
