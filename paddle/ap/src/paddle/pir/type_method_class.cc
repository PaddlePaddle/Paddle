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

#include "paddle/ap/include/paddle/pir/type_method_class.h"
#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/value_method_class.h"
#include "paddle/ap/include/paddle/pir/type_adt_type_id.h"

namespace ap::paddle {

adt::Result<axpr::Value> PirTypeString(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(self, self_val.template CastTo<pir::Type>());
  std::ostringstream ss;
  ss << self;
  return ss.str();
}

struct PirTypeGetType {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<pir::Type>());
    const auto& type_id = GetTypeAdtTypeId(self);
    return type_id.Match([&](const auto& impl) -> std::string {
      using T = typename std::decay_t<decltype(impl)>::type;
      return T::name();
    });
  }
};

struct ConvertToDtype {
  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<pir::Type>());
    try {
      auto phi_type = ::paddle::dialect::TransToPhiDataType(self);
      ADT_LET_CONST_REF(dtype, axpr::GetDataTypeFromPhiDataType(phi_type));
      return dtype;
    } catch (const std::exception& e) {
      return adt::errors::ValueError{e.what()};
    }
  }
};

struct PirTypeMatch {
  static adt::Result<axpr::Value> Call(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& packed_args_val) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<pir::Type>());
    const auto& type_id = GetTypeAdtTypeId(self);
    const auto& type_name = type_id.Match([&](const auto& impl) -> std::string {
      using T = typename std::decay_t<decltype(impl)>::type;
      return T::name();
    });
    const auto& packed_args =
        axpr::CastToPackedArgs<axpr::Value>(packed_args_val);
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->size() == 0) << adt::errors::TypeError{
        std::string() +
        "PirType.match() supports keyword arguments only, but " +
        std::to_string(args->size()) + " positional arguments were given"};
    auto PatternMatch =
        [&](const auto& impl) -> adt::Result<adt::List<axpr::Value>> {
      using T = typename std::decay_t<decltype(impl)>::type;
      return MakePirTypeImpl<T>::GetCallArgs(self_val);
    };
    std::string key = type_name;
    if (!kwargs->Has(type_name)) {
      if (!kwargs->Has("_")) {
        return adt::errors::TypeError{std::string() +
                                      "PirType.match() failed. no keyword '" +
                                      type_name + "' or '_' provided"};
      }
      key = "_";
    }
    ADT_LET_CONST_REF(func, kwargs->Get(key));
    ADT_LET_CONST_REF(type_make_args, type_id.Match(PatternMatch));
    ADT_CHECK(axpr::CallableHelper{}.IsCallable(func))
        << adt::errors::TypeError{
               std::string() +
               "the arguments of PirType.match() should be callable"};
    if (key == "_") {
      return interpreter->InterpretCall(func, {});
    } else {
      return interpreter->InterpretCall(func, type_make_args.vector());
    }
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetPirTypeClass() {
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("PirType", [&](const auto& Yield) {
        Yield("__str__", &PirTypeString);
        Yield("get_type_name", &PirTypeGetType::Call);
        Yield("convert_to_dtype", &ConvertToDtype::Call);
        Yield("match", &PirTypeMatch::Call);
      }));
  return axpr::MakeGlobalNaiveClassOps<pir::Type>(cls);
}

adt::Result<axpr::Value> MakePirTypeImplNullType::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  pir::Type type;
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplNullType::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplVectorType::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
      std::string() + "pir.t_vec() takes 1 argument, but " +
      std::to_string(args.size()) + " were given"};
  ADT_LET_CONST_REF(lst, args.at(0).template CastTo<adt::List<axpr::Value>>())
      << adt::errors::TypeError{
             std::string() +
             "the argument 1 of pir.t_vec() should be a list (not a " +
             axpr::GetTypeName(args.at(0)) + ")"};
  std::vector<pir::Type> types;
  for (const auto& arg : *lst) {
    ADT_LET_CONST_REF(elt, arg.template CastTo<pir::Type>());
    types.emplace_back(elt);
  }
  const pir::Type type{pir::VectorType::get(pir::IrContext::Instance(), types)};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplVectorType::GetCallArgs(
    const axpr::Value& self_val) {
  ADT_LET_CONST_REF(pir_type, self_val.template CastTo<pir::Type>());
  ADT_CHECK(pir_type.isa<pir::VectorType>());
  const auto& type_list = pir_type.dyn_cast<pir::VectorType>();
  adt::List<axpr::Value> ret_list{};
  ret_list->reserve(type_list.size());
  for (int i = 0; i < type_list.size(); ++i) {
    ret_list->emplace_back(GetPirTypeClass().New(type_list[i]));
  }
  return adt::List<axpr::Value>{axpr::Value{ret_list}};
}

adt::Result<axpr::Value> MakePirTypeImplDenseTensorType::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 3);
  ADT_LET_CONST_REF(type, args.at(0).template CastTo<pir::Type>());
  ADT_LET_CONST_REF(int_list,
                    args.at(1).template CastTo<adt::List<axpr::Value>>());
  std::vector<int64_t> dims;
  dims.reserve(int_list->size());
  for (const auto& int_val : *int_list) {
    ADT_LET_CONST_REF(elt, int_val.template CastTo<int64_t>());
    dims.emplace_back(elt);
  }
  ::common::DDim ddim(dims.data(), dims.size());
  ADT_LET_CONST_REF(data_layout_str, args.at(2).template CastTo<std::string>());
  std::optional<::common::DataLayout> data_layout;
  try {
    data_layout = ::common::StringToDataLayout(data_layout_str);
  } catch (const std::exception&) {
    return adt::errors::ValueError{"StringToDataLayout('" + data_layout_str +
                                   "') failed"};
  }
  ADT_CHECK(data_layout.has_value());
  const pir::Type dense_tensor_type{pir::DenseTensorType::get(
      pir::IrContext::Instance(), type, ddim, data_layout.value())};
  return GetPirTypeClass().New(dense_tensor_type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplDenseTensorType::GetCallArgs(
    const axpr::Value& self_val) {
  ADT_LET_CONST_REF(pir_type, self_val.template CastTo<pir::Type>());
  ADT_CHECK(pir_type.isa<pir::DenseTensorType>());
  const auto& dense_tensor_type = pir_type.dyn_cast<pir::DenseTensorType>();
  // dtype
  const auto& dtype = GetPirTypeClass().New(dense_tensor_type.dtype());
  // shape
  adt::List<axpr::Value> dims{};
  dims->reserve(dense_tensor_type.dims().size());
  for (int i = 0; i < dense_tensor_type.dims().size(); ++i) {
    int64_t dim = dense_tensor_type.dims().at(i);
    dims->emplace_back(dim);
  }
  // data layout
  std::string data_layout_str;
  try {
    data_layout_str =
        ::common::DataLayoutToString(dense_tensor_type.data_layout());
  } catch (const std::exception& e) {
    return adt::errors::ValueError{e.what()};
  }
  return adt::List<axpr::Value>{dtype, dims, data_layout_str};
}

adt::Result<axpr::Value> MakePirTypeImplBFloat16Type::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::BFloat16Type::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplBFloat16Type::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplFloat16Type::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::Float16Type::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplFloat16Type::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplFloat32Type::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::Float32Type::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplFloat32Type::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplFloat64Type::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::Float64Type::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplFloat64Type::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplInt8Type::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::Int8Type::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplInt8Type::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplUInt8Type::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::UInt8Type::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplUInt8Type::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplInt16Type::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::Int16Type::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplInt16Type::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplInt32Type::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::Int32Type::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplInt32Type::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplInt64Type::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::Int64Type::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplInt64Type::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplIndexType::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::IndexType::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplIndexType::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplBoolType::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::BoolType::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplBoolType::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplComplex64Type::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::Complex64Type::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplComplex64Type::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplComplex128Type::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  const pir::Type type{pir::Complex128Type::get(pir::IrContext::Instance())};
  return GetPirTypeClass().New(type);
}

adt::Result<adt::List<axpr::Value>> MakePirTypeImplComplex128Type::GetCallArgs(
    const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplSelectedRowsType::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 3);
  ADT_LET_CONST_REF(type, args.at(0).template CastTo<pir::Type>());
  ADT_LET_CONST_REF(int_list,
                    args.at(1).template CastTo<adt::List<axpr::Value>>());
  std::vector<int64_t> dims;
  dims.reserve(int_list->size());
  for (const auto& int_val : *int_list) {
    ADT_LET_CONST_REF(elt, int_val.template CastTo<int64_t>());
    dims.emplace_back(elt);
  }
  ::common::DDim ddim(dims.data(), dims.size());
  ADT_LET_CONST_REF(data_layout_str, args.at(2).template CastTo<std::string>());
  std::optional<::common::DataLayout> data_layout;
  try {
    data_layout = ::common::StringToDataLayout(data_layout_str);
  } catch (const std::exception&) {
    return adt::errors::ValueError{"StringToDataLayout('" + data_layout_str +
                                   "') failed"};
  }
  ADT_CHECK(data_layout.has_value());
  const pir::Type pir_type{::paddle::dialect::SelectedRowsType::get(
      pir::IrContext::Instance(), type, ddim, data_layout.value())};
  return GetPirTypeClass().New(pir_type);
}

adt::Result<adt::List<axpr::Value>>
MakePirTypeImplSelectedRowsType::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(
      pir_type,
      self_val.template CastTo<::paddle::dialect::SelectedRowsType>());
  // dtype
  const auto& dtype = GetPirTypeClass().New(pir_type.dtype());
  // shape
  adt::List<axpr::Value> dims{};
  dims->reserve(pir_type.dims().size());
  for (int i = 0; i < pir_type.dims().size(); ++i) {
    int64_t dim = pir_type.dims().at(i);
    dims->emplace_back(dim);
  }
  // data layout
  std::string data_layout_str;
  try {
    data_layout_str = ::common::DataLayoutToString(pir_type.data_layout());
  } catch (const std::exception& e) {
    return adt::errors::ValueError{e.what()};
  }
  return adt::List<axpr::Value>{dtype, dims, data_layout_str};
}

adt::Result<axpr::Value> MakePirTypeImplDenseTensorArrayType::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 3);
  ADT_LET_CONST_REF(type, args.at(0).template CastTo<pir::Type>());
  ADT_LET_CONST_REF(int_list,
                    args.at(1).template CastTo<adt::List<axpr::Value>>());
  std::vector<int64_t> dims;
  dims.reserve(int_list->size());
  for (const auto& int_val : *int_list) {
    ADT_LET_CONST_REF(elt, int_val.template CastTo<int64_t>());
    dims.emplace_back(elt);
  }
  ::common::DDim ddim(dims.data(), dims.size());
  ADT_LET_CONST_REF(data_layout_str, args.at(2).template CastTo<std::string>());
  std::optional<::common::DataLayout> data_layout;
  try {
    data_layout = ::common::StringToDataLayout(data_layout_str);
  } catch (const std::exception&) {
    return adt::errors::ValueError{"StringToDataLayout('" + data_layout_str +
                                   "') failed"};
  }
  ADT_CHECK(data_layout.has_value());
  const pir::Type dense_tensor_type{
      ::paddle::dialect::DenseTensorArrayType::get(
          pir::IrContext::Instance(), type, ddim, data_layout.value())};
  return GetPirTypeClass().New(dense_tensor_type);
}

adt::Result<adt::List<axpr::Value>>
MakePirTypeImplDenseTensorArrayType::GetCallArgs(const axpr::Value& self_val) {
  ADT_LET_CONST_REF(
      dense_tensor_array_type,
      self_val.template CastTo<::paddle::dialect::DenseTensorArrayType>());
  // dtype
  const auto& dtype = GetPirTypeClass().New(dense_tensor_array_type.dtype());
  // shape
  adt::List<axpr::Value> dims{};
  dims->reserve(dense_tensor_array_type.dims().size());
  for (int i = 0; i < dense_tensor_array_type.dims().size(); ++i) {
    int64_t dim = dense_tensor_array_type.dims().at(i);
    dims->emplace_back(dim);
  }
  // data layout
  std::string data_layout_str;
  try {
    data_layout_str =
        ::common::DataLayoutToString(dense_tensor_array_type.data_layout());
  } catch (const std::exception& e) {
    return adt::errors::ValueError{e.what()};
  }
  return adt::List<axpr::Value>{dtype, dims, data_layout_str};
}

adt::Result<axpr::Value> MakePirTypeImplSparseCooTensorType::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  return adt::errors::NotImplementedError{
      std::string() + ::paddle::dialect::SparseCooTensorType::name() +
      "() is not implemented"};
}

adt::Result<adt::List<axpr::Value>>
MakePirTypeImplSparseCooTensorType::GetCallArgs(const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplSparseCsrTensorType::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  return adt::errors::NotImplementedError{
      std::string() + ::paddle::dialect::SparseCsrTensorType::name() +
      "() is not implemented"};
}

adt::Result<adt::List<axpr::Value>>
MakePirTypeImplSparseCsrTensorType::GetCallArgs(const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

adt::Result<axpr::Value> MakePirTypeImplUnclassifiedType::Call(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  return adt::errors::NotImplementedError{
      std::string() + UnclassifiedType::name() + "() is not implemented"};
}

adt::Result<adt::List<axpr::Value>>
MakePirTypeImplUnclassifiedType::GetCallArgs(const axpr::Value& self_val) {
  return adt::List<axpr::Value>{};
}

}  // namespace ap::paddle
