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

#include "paddle/ap/include/paddle/pir/attribute_method_class.h"
#include "paddle/ap/include/axpr/abstract_list.h"
#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/paddle/phi/place_method_class.h"
#include "paddle/ap/include/paddle/pir/shape_or_data_method_class.h"
#include "paddle/ap/include/paddle/pir/type_method_class.h"

namespace ap::paddle {

inline adt::Result<axpr::Value> PirAttributeToString(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  ADT_LET_CONST_REF(self, self_val.template CastTo<pir::Attribute>());
  std::ostringstream ss;
  ss << self;
  return ss.str();
}

struct PirAttributeGetType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<pir::Attribute>());
    const auto& attr_type_id = GetAttrAdtTypeId(self);
    return attr_type_id.Match([&](const auto& impl) -> std::string {
      using T = typename std::decay_t<decltype(impl)>::type;
      return T::name();
    });
  }
};

struct PirAttributeMatch {
  static adt::Result<axpr::Value> Call(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& packed_args_val) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<pir::Attribute>());
    const auto& attr_type_id = GetAttrAdtTypeId(self);
    const auto& type_name =
        attr_type_id.Match([&](const auto& impl) -> std::string {
          using T = typename std::decay_t<decltype(impl)>::type;
          return T::name();
        });
    const auto& packed_args =
        axpr::CastToPackedArgs<axpr::Value>(packed_args_val);
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->size() == 0) << adt::errors::TypeError{
        std::string() +
        "PirAttribute.match() supports keyword arguments only, but " +
        std::to_string(args->size()) + " positional arguments were given"};
    std::string key = type_name;
    if (!kwargs->Has(type_name)) {
      if (!kwargs->Has("_")) {
        return adt::errors::TypeError{
            std::string() + "PirAttribute.match() failed. no keyword '" +
            type_name + "' or '_' provided"};
      }
      key = "_";
    }
    ADT_LET_CONST_REF(func, kwargs->Get(key));
    ADT_CHECK(axpr::CallableHelper{}.IsCallable(func))
        << adt::errors::TypeError{
               std::string() +
               "the arguments of PirAttribute.match() should be callable"};
    if (key == "_") {
      return interpreter->InterpretCall(func, {});
    } else {
      auto PatternMatch =
          [&](const auto& impl) -> adt::Result<adt::List<axpr::Value>> {
        using T = typename std::decay_t<decltype(impl)>::type;
        return MakePirAttributeImpl<T>::GetCallArgs(self_val);
      };
      ADT_LET_CONST_REF(attr_make_args, attr_type_id.Match(PatternMatch));
      return interpreter->InterpretCall(func, attr_make_args.vector());
    }
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetPirAttributeClass() {
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "PirAttribute", [&](const auto& Yield) {
        Yield("__str__", &PirAttributeToString);
        Yield("get_type", &PirAttributeGetType::Call);
        Yield("match", &PirAttributeMatch::Call);
      }));
  return axpr::MakeGlobalNaiveClassOps<pir::Attribute>(cls);
}

adt::Result<axpr::Value> MakePirAttributeImplBoolAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(bool_val, args.at(0).template CastTo<bool>());
  pir::Attribute attr{
      pir::BoolAttribute::get(pir::IrContext::Instance(), bool_val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplBoolAttribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::BoolAttribute>());
  const auto& attr = attribute.dyn_cast<pir::BoolAttribute>();
  axpr::Value val{attr.data()};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplComplex64Attribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
  ADT_LET_CONST_REF(complex_val, data_val.template TryGet<axpr::complex64>());
  pir::Attribute attr{
      pir::Complex64Attribute::get(pir::IrContext::Instance(), complex_val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplComplex64Attribute::GetCallArgs(
    const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::Complex64Attribute>());
  const auto& attr = attribute.dyn_cast<pir::Complex64Attribute>();
  axpr::Value val{axpr::DataValue{attr.data()}};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplComplex128Attribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
  ADT_LET_CONST_REF(val, data_val.template TryGet<axpr::complex128>());
  pir::Attribute attr{
      pir::Complex128Attribute::get(pir::IrContext::Instance(), val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplComplex128Attribute::GetCallArgs(
    const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::Complex128Attribute>());
  const auto& attr = attribute.dyn_cast<pir::Complex128Attribute>();
  axpr::Value val{axpr::DataValue{attr.data()}};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplFloatAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
  ADT_LET_CONST_REF(val, data_val.template TryGet<float>());
  pir::Attribute attr{
      pir::FloatAttribute::get(pir::IrContext::Instance(), val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplFloatAttribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::FloatAttribute>());
  const auto& attr = attribute.dyn_cast<pir::FloatAttribute>();
  axpr::Value val{axpr::DataValue{attr.data()}};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplDoubleAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
  ADT_LET_CONST_REF(val, data_val.template TryGet<double>());
  pir::Attribute attr{
      pir::DoubleAttribute::get(pir::IrContext::Instance(), val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplDoubleAttribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::DoubleAttribute>());
  const auto& attr = attribute.dyn_cast<pir::DoubleAttribute>();
  axpr::Value val{axpr::DataValue{attr.data()}};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplInt32Attribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
  ADT_LET_CONST_REF(val, data_val.template TryGet<int32_t>());
  pir::Attribute attr{
      pir::Int32Attribute::get(pir::IrContext::Instance(), val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplInt32Attribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::Int32Attribute>());
  const auto& attr = attribute.dyn_cast<pir::Int32Attribute>();
  axpr::Value val{axpr::DataValue{attr.data()}};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplIndexAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
  ADT_LET_CONST_REF(val, data_val.template TryGet<int64_t>());
  pir::Attribute attr{
      pir::IndexAttribute::get(pir::IrContext::Instance(), val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplIndexAttribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::IndexAttribute>());
  const auto& attr = attribute.dyn_cast<pir::IndexAttribute>();
  axpr::Value val{axpr::DataValue{attr.data()}};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplInt64Attribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
  ADT_LET_CONST_REF(val, data_val.template TryGet<int64_t>());
  pir::Attribute attr{
      pir::Int64Attribute::get(pir::IrContext::Instance(), val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplInt64Attribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::Int64Attribute>());
  const auto& attr = attribute.dyn_cast<pir::Int64Attribute>();
  axpr::Value val{axpr::DataValue{attr.data()}};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplPointerAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::PointerValue>());
  ADT_LET_CONST_REF(val, data_val.template TryGet<void*>());
  pir::Attribute attr{
      pir::PointerAttribute::get(pir::IrContext::Instance(), val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplPointerAttribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::PointerAttribute>());
  const auto& attr = attribute.dyn_cast<pir::PointerAttribute>();
  axpr::Value val{axpr::PointerValue{attr.data()}};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplTypeAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(type_val, args.at(0).template CastTo<pir::Type>());
  pir::Attribute attr{
      pir::TypeAttribute::get(pir::IrContext::Instance(), type_val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplTypeAttribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::TypeAttribute>());
  const auto& attr = attribute.dyn_cast<pir::TypeAttribute>();
  axpr::Value val{GetPirTypeClass().New(attr.data())};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplStrAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(val, args.at(0).template CastTo<std::string>());
  pir::Attribute attr{pir::StrAttribute::get(pir::IrContext::Instance(), val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplStrAttribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::StrAttribute>());
  const auto& attr = attribute.dyn_cast<pir::StrAttribute>();
  axpr::Value val{attr.AsString()};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplArrayAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
      std::string() + "pir.t_vec() takes 1 argument, but " +
      std::to_string(args.size()) + " were given"};
  ADT_LET_CONST_REF(lst, args.at(0).template CastTo<adt::List<axpr::Value>>())
      << adt::errors::TypeError{
             std::string() +
             "the argument of pir.t_vec() should be a list (not a " +
             axpr::GetTypeName(args.at(0)) + ")"};
  std::vector<pir::Attribute> attrs;
  attrs.reserve(lst->size());
  for (const auto& arg : *lst) {
    ADT_LET_CONST_REF(elt, arg.template CastTo<pir::Attribute>());
    attrs.emplace_back(elt);
  }
  pir::Attribute attr{
      pir::ArrayAttribute::get(pir::IrContext::Instance(), attrs)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplArrayAttribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::ArrayAttribute>());
  const auto& attr = attribute.dyn_cast<pir::ArrayAttribute>();
  adt::List<axpr::Value> lst{};
  for (int i = 0; i < attr.size(); ++i) {
    lst->emplace_back(GetPirAttributeClass().New(attr.at(i)));
  }
  return adt::List<axpr::Value>{axpr::Value{lst}};
}

adt::Result<axpr::Value> MakePirAttributeImplTensorNameAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(val, args.at(0).template CastTo<std::string>());
  pir::Attribute attr{
      pir::TensorNameAttribute::get(pir::IrContext::Instance(), val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplTensorNameAttribute::GetCallArgs(
    const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::TensorNameAttribute>());
  const auto& attr = attribute.dyn_cast<pir::TensorNameAttribute>();
  axpr::Value val{attr.data()};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplSymbolAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(shape_or_data,
                    args.at(0).template CastTo<symbol::ShapeOrDataDimExprs>());
  pir::Attribute attr{pir::shape::SymbolAttribute::get(
      pir::IrContext::Instance(), shape_or_data)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplSymbolAttribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<pir::shape::SymbolAttribute>());
  const auto& attr = attribute.dyn_cast<pir::shape::SymbolAttribute>();
  axpr::Value val{GetPirShapeOrDataClass().New(attr.data())};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplKernelAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  return adt::errors::NotImplementedError{
      std::string() + "pir." + ::paddle::dialect::KernelAttribute::name() +
      "() is not implemneted"};
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplKernelAttribute::GetCallArgs(const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirAttributeImplIntArrayAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
      std::string() + ::paddle::dialect::IntArrayAttribute::name() +
      "() takes 1 argument, but " + std::to_string(args.size()) +
      " were given"};
  ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(args.at(0)))
      << adt::errors::TypeError{
             std::string() + "the argument 1 of" +
             ::paddle::dialect::IntArrayAttribute::name() +
             "() should be a list/SerializableList/MutableList (not " +
             axpr::GetTypeName(args.at(0)) + ")"};
  std::vector<int64_t> int_array;
  ADT_LET_CONST_REF(lst_size, lst.size());
  int_array.reserve(lst_size);
  ADT_RETURN_IF_ERR(
      lst.Visit([&](const auto& arg) -> adt::Result<adt::LoopCtrl> {
        ADT_LET_CONST_REF(elt, arg.template CastTo<int64_t>());
        int_array.emplace_back(elt);
        return adt::Continue{};
      }));
  pir::Attribute attr{::paddle::dialect::IntArrayAttribute::get(
      pir::IrContext::Instance(), int_array)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplIntArrayAttribute::GetCallArgs(
    const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<::paddle::dialect::IntArrayAttribute>());
  const auto& attr = attribute.dyn_cast<::paddle::dialect::IntArrayAttribute>();
  adt::List<axpr::Value> lst{};
  const auto& data = attr.data();
  for (int i = 0; i < data.size(); ++i) {
    int64_t elt = data[i];
    lst->emplace_back(elt);
  }
  return adt::List<axpr::Value>{axpr::Value{lst}};
}

inline adt::Result<phi::Scalar> ConvertDataValueToScalar(
    const axpr::DataValue& data_val) {
  return ScalarHelper{}.ConvertFromDataType(data_val);
}

adt::Result<axpr::Value> MakePirAttributeImplScalarAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(data_val, args.at(0).template CastTo<axpr::DataValue>());
  ADT_LET_CONST_REF(val, ConvertDataValueToScalar(data_val));
  pir::Attribute attr{
      ::paddle::dialect::ScalarAttribute::get(pir::IrContext::Instance(), val)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplScalarAttribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<::paddle::dialect::ScalarAttribute>());
  const auto& attr = attribute.dyn_cast<::paddle::dialect::ScalarAttribute>();
  ADT_LET_CONST_REF(data_value, ScalarHelper{}.ConvertToDataValue(attr.data()));
  axpr::Value val{data_value};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplDataTypeAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  std::optional<phi::DataType> opt_phi_data_type;
  if (args.at(0).template CastableTo<pir::Type>()) {
    ADT_LET_CONST_REF(type, args.at(0).template CastTo<pir::Type>());
    opt_phi_data_type = ::paddle::dialect::TransToPhiDataType(type);
  } else if (args.at(0).template CastableTo<axpr::DataType>()) {
    ADT_LET_CONST_REF(data_type, args.at(0).template CastTo<axpr::DataType>());
    ADT_LET_CONST_REF(phi_data_type,
                      axpr::GetPhiDataTypeFromDataType(data_type));
    opt_phi_data_type = phi_data_type;
  } else {
    return adt::errors::TypeError{
        "the argument 1 of t_dtype() should be a DataType/PirType (not a " +
        axpr::GetTypeName(args.at(0)) + ")"};
  }
  pir::Attribute attr{::paddle::dialect::DataTypeAttribute::get(
      pir::IrContext::Instance(), opt_phi_data_type.value())};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplDataTypeAttribute::GetCallArgs(
    const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<::paddle::dialect::DataTypeAttribute>());
  const auto& attr = attribute.dyn_cast<::paddle::dialect::DataTypeAttribute>();
  ADT_LET_CONST_REF(data_type, axpr::GetDataTypeFromPhiDataType(attr.data()));
  axpr::Value val{data_type};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplPlaceAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(place, args.at(0).template CastTo<phi::Place>());
  pir::Attribute attr{::paddle::dialect::PlaceAttribute::get(
      pir::IrContext::Instance(), place)};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplPlaceAttribute::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<::paddle::dialect::PlaceAttribute>());
  const auto& attr = attribute.dyn_cast<::paddle::dialect::PlaceAttribute>();
  axpr::Value val{GetPlaceClass().New(attr.data())};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplDataLayoutAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(data_layout_str, args.at(0).template CastTo<std::string>());
  std::optional<::common::DataLayout> data_layout;
  try {
    data_layout = ::common::StringToDataLayout(data_layout_str);
  } catch (const std::exception&) {
    return adt::errors::ValueError{"StringToDataLayout('" + data_layout_str +
                                   "') failed"};
  }
  pir::Attribute attr{::paddle::dialect::DataLayoutAttribute::get(
      pir::IrContext::Instance(), data_layout.value())};
  return GetPirAttributeClass().New(attr);
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplDataLayoutAttribute::GetCallArgs(
    const axpr::Value& self_val) {
  ADT_LET_CONST_REF(attribute, self_val.template CastTo<pir::Attribute>());
  ADT_CHECK(attribute.isa<::paddle::dialect::DataLayoutAttribute>());
  const auto& attr =
      attribute.dyn_cast<::paddle::dialect::DataLayoutAttribute>();
  std::string data_layout_str;
  try {
    data_layout_str = ::common::DataLayoutToString(attr.data());
  } catch (const std::exception& e) {
    return adt::errors::ValueError{e.what()};
  }
  axpr::Value val{data_layout_str};
  return adt::List<axpr::Value>{val};
}

adt::Result<axpr::Value> MakePirAttributeImplGroupInfoAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  return adt::errors::NotImplementedError{
      std::string() + "pir." + ::cinn::dialect::GroupInfoAttribute::name() +
      "() is not implemneted"};
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplGroupInfoAttribute::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirAttributeImplCINNKernelInfoAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  return adt::errors::NotImplementedError{
      std::string() + "pir." +
      ::cinn::dialect::CINNKernelInfoAttribute::name() +
      "() is not implemneted"};
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplCINNKernelInfoAttribute::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirAttributeImplUnclassifiedAttribute::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  return adt::errors::NotImplementedError{std::string() + "pir." +
                                          UnclassifiedAttribute::name() +
                                          "() is not implemneted"};
}

adt::Result<adt::List<axpr::Value>>
MakePirAttributeImplUnclassifiedAttribute::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

}  // namespace ap::paddle
