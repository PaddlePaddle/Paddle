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
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/paddle/pir/type.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"

namespace ap::paddle {

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetPirTypeClass();

template <typename T>
struct MakePirTypeImpl;

struct MakePirTypeImplNullType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<NullType> : public MakePirTypeImplNullType {};

struct MakePirTypeImplVectorType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::VectorType> : public MakePirTypeImplVectorType {};

struct MakePirTypeImplDenseTensorType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::DenseTensorType>
    : public MakePirTypeImplDenseTensorType {};

struct MakePirTypeImplBFloat16Type {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::BFloat16Type>
    : public MakePirTypeImplBFloat16Type {};

struct MakePirTypeImplFloat16Type {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::Float16Type> : public MakePirTypeImplFloat16Type {
};

struct MakePirTypeImplFloat32Type {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::Float32Type> : public MakePirTypeImplFloat32Type {
};

struct MakePirTypeImplFloat64Type {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::Float64Type> : public MakePirTypeImplFloat64Type {
};

struct MakePirTypeImplInt8Type {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::Int8Type> : public MakePirTypeImplInt8Type {};

struct MakePirTypeImplUInt8Type {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::UInt8Type> : public MakePirTypeImplUInt8Type {};

struct MakePirTypeImplInt16Type {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::Int16Type> : public MakePirTypeImplInt16Type {};

struct MakePirTypeImplInt32Type {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::Int32Type> : public MakePirTypeImplInt32Type {};

struct MakePirTypeImplInt64Type {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::Int64Type> : public MakePirTypeImplInt64Type {};

struct MakePirTypeImplIndexType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::IndexType> : public MakePirTypeImplIndexType {};

struct MakePirTypeImplBoolType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::BoolType> : public MakePirTypeImplBoolType {};

struct MakePirTypeImplComplex64Type {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::Complex64Type>
    : public MakePirTypeImplComplex64Type {};

struct MakePirTypeImplComplex128Type {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::pir::Complex128Type>
    : public MakePirTypeImplComplex128Type {};

struct MakePirTypeImplSelectedRowsType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::paddle::dialect::SelectedRowsType>
    : public MakePirTypeImplSelectedRowsType {};

struct MakePirTypeImplDenseTensorArrayType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::paddle::dialect::DenseTensorArrayType>
    : public MakePirTypeImplDenseTensorArrayType {};

struct MakePirTypeImplSparseCooTensorType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::paddle::dialect::SparseCooTensorType>
    : public MakePirTypeImplSparseCooTensorType {};

struct MakePirTypeImplSparseCsrTensorType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<::paddle::dialect::SparseCsrTensorType>
    : public MakePirTypeImplSparseCsrTensorType {};

struct MakePirTypeImplUnclassifiedType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args);
  static adt::Result<adt::List<axpr::Value>> GetCallArgs(
      const axpr::Value& self_val);
};
template <>
struct MakePirTypeImpl<UnclassifiedType>
    : public MakePirTypeImplUnclassifiedType {};

}  // namespace ap::paddle
