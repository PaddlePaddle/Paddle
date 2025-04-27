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

#include "paddle/ap/include/paddle/pir/pir_to_anf_expr_helper.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/drr/value_method_class.h"
#include "paddle/ap/include/paddle/phi/scalar_helper.h"

namespace ap::paddle {

namespace {

template <typename TypeIdT>
struct TypeToAnfExprConverter;

template <>
struct TypeToAnfExprConverter<NullType> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(NullType::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::VectorType> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    ADT_CHECK(type.template isa<::pir::VectorType>());
    auto vec_type = type.template dyn_cast<::pir::VectorType>();
    std::vector<axpr::AnfExpr> args;
    args.reserve(vec_type.size());
    for (const auto& elt_type : vec_type.data()) {
      ADT_LET_CONST_REF(
          elt_anf_expr,
          PirToAnfExprHelper{}.ConvertPirTypeToAnfExpr(ctx, elt_type));
      args.emplace_back(elt_anf_expr);
    }
    return ctx->Var("pir")
        .Attr(::pir::VectorType::name())
        .Call(ctx->Var(axpr::kBuiltinList()).Apply(args));
  }
};

adt::Result<axpr::AnfExpr> ConvertToDataLayoutAnfExpr(
    axpr::LetContext* ctx, const ::common::DataLayout& data_layout) {
  try {
    const auto& data_layout_str = ::common::DataLayoutToString(data_layout);
    return ctx->String(data_layout_str);
  } catch (const std::exception& e) {
    return adt::errors::ValueError{e.what()};
  }
}

template <>
struct TypeToAnfExprConverter<::pir::DenseTensorType> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    ADT_CHECK(type.template isa<::pir::DenseTensorType>());
    auto dense_tensor_type = type.template dyn_cast<::pir::DenseTensorType>();

    // dtype
    ADT_LET_CONST_REF(dtype_anf_expr,
                      PirToAnfExprHelper{}.ConvertPirTypeToAnfExpr(
                          ctx, dense_tensor_type.dtype()));

    // dims
    std::vector<axpr::AnfExpr> dim_elts;
    const auto& dims = dense_tensor_type.dims();
    for (int i = 0; i < dims.size(); ++i) {
      dim_elts.push_back(ctx->Int64(dims.at(i)));
    }
    const auto& dims_anf_expr = ctx->Var(axpr::kBuiltinList()).Apply(dim_elts);

    // data layout
    ADT_LET_CONST_REF(
        data_layout_anf_expr,
        ConvertToDataLayoutAnfExpr(ctx, dense_tensor_type.data_layout()));
    return ctx->Var("pir")
        .Attr(::pir::DenseTensorType::name())
        .Call(dtype_anf_expr, dims_anf_expr, data_layout_anf_expr);
  }
};

template <>
struct TypeToAnfExprConverter<::pir::BFloat16Type> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::BFloat16Type::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::Float16Type> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::Float16Type::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::Float32Type> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::Float32Type::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::Float64Type> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::Float64Type::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::Int8Type> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::Int8Type::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::UInt8Type> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::UInt8Type::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::Int16Type> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::Int16Type::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::Int32Type> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::Int32Type::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::Int64Type> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::Int64Type::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::IndexType> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::IndexType::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::BoolType> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::BoolType::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::Complex64Type> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::Complex64Type::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::pir::Complex128Type> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir").Attr(::pir::Complex128Type::name()).Call();
  }
};

template <>
struct TypeToAnfExprConverter<::paddle::dialect::SelectedRowsType> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return ctx->Var("pir")
        .Attr(::paddle::dialect::SelectedRowsType::name())
        .Call();
  }
};

template <>
struct TypeToAnfExprConverter<::paddle::dialect::DenseTensorArrayType> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    ADT_CHECK(type.template isa<::paddle::dialect::DenseTensorArrayType>());
    auto dense_tensor_array_type =
        type.template dyn_cast<::paddle::dialect::DenseTensorArrayType>();

    // dtype
    ADT_LET_CONST_REF(dtype_anf_expr,
                      PirToAnfExprHelper{}.ConvertPirTypeToAnfExpr(
                          ctx, dense_tensor_array_type.dtype()));

    // dims
    std::vector<axpr::AnfExpr> dim_elts;
    const auto& dims = dense_tensor_array_type.dims();
    for (int i = 0; i < dims.size(); ++i) {
      dim_elts.push_back(ctx->Int64(dims.at(i)));
    }
    const auto& dims_anf_expr = ctx->Var(axpr::kBuiltinList()).Apply(dim_elts);

    // data layout
    ADT_LET_CONST_REF(
        data_layout_anf_expr,
        ConvertToDataLayoutAnfExpr(ctx, dense_tensor_array_type.data_layout()));
    return ctx->Var("pir")
        .Attr(::paddle::dialect::DenseTensorArrayType::name())
        .Call(dtype_anf_expr, dims_anf_expr, data_layout_anf_expr);
  }
};

template <>
struct TypeToAnfExprConverter<::paddle::dialect::SparseCooTensorType> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return adt::errors::NotImplementedError{
        std::string() + ::paddle::dialect::SparseCooTensorType::name() +
        "() is not implemented"};
  }
};

template <>
struct TypeToAnfExprConverter<::paddle::dialect::SparseCsrTensorType> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return adt::errors::NotImplementedError{
        std::string() + ::paddle::dialect::SparseCsrTensorType::name() +
        "() is not implemented"};
  }
};

template <>
struct TypeToAnfExprConverter<UnclassifiedType> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Type type) {
    return adt::errors::NotImplementedError{
        std::string() + UnclassifiedType::name() + "() is not implemented"};
  }
};

template <typename T>
struct AttrToAnfExprConverter;

template <>
struct AttrToAnfExprConverter<pir::BoolAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::BoolAttribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::BoolAttribute>();
    const auto& attr_data_val = ctx->Bool(attr_impl.data());
    return ctx->Var("pir").Attr(pir::BoolAttribute::name()).Call(attr_data_val);
  }
};

adt::Result<axpr::AnfExpr> ConvertToComplex64AnfExpr(
    axpr::LetContext* ctx, const axpr::complex64& attr_data) {
  const auto& real = ctx->Var("DataValue")
                         .Attr("float32")
                         .Call(ctx->String(std::to_string(attr_data.real)));
  const auto& imag = ctx->Var("DataValue")
                         .Attr("float32")
                         .Call(ctx->String(std::to_string(attr_data.imag)));
  const auto& attr_data_val =
      ctx->Var("DataValue").Attr("complex64").Call(real, imag);
  return attr_data_val;
}

template <>
struct AttrToAnfExprConverter<pir::Complex64Attribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::Complex64Attribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::Complex64Attribute>();
    const auto& attr_data = attr_impl.data();
    ADT_LET_CONST_REF(attr_data_val, ConvertToComplex64AnfExpr(ctx, attr_data));
    return ctx->Var("pir")
        .Attr(pir::Complex64Attribute::name())
        .Call(attr_data_val);
  }
};

adt::Result<axpr::AnfExpr> ConvertToComplex128AnfExpr(
    axpr::LetContext* ctx, const axpr::complex128& attr_data) {
  const auto& real = ctx->Var("DataValue")
                         .Attr("float64")
                         .Call(ctx->String(std::to_string(attr_data.real)));
  const auto& imag = ctx->Var("DataValue")
                         .Attr("float64")
                         .Call(ctx->String(std::to_string(attr_data.imag)));
  const auto& attr_data_val =
      ctx->Var("DataValue").Attr("complex128").Call(real, imag);
  return attr_data_val;
}

template <>
struct AttrToAnfExprConverter<pir::Complex128Attribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::Complex128Attribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::Complex128Attribute>();
    const auto& attr_data = attr_impl.data();
    ADT_LET_CONST_REF(attr_data_val,
                      ConvertToComplex128AnfExpr(ctx, attr_data));
    return ctx->Var("pir")
        .Attr(pir::Complex128Attribute::name())
        .Call(attr_data_val);
  }
};

template <>
struct AttrToAnfExprConverter<pir::FloatAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::FloatAttribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::FloatAttribute>();
    const auto& attr_str = ctx->String(std::to_string(attr_impl.data()));
    const auto& attr_data_val =
        ctx->Var("DataValue").Attr("float32").Call(attr_str);
    return ctx->Var("pir")
        .Attr(pir::FloatAttribute::name())
        .Call(attr_data_val);
  }
};

template <>
struct AttrToAnfExprConverter<pir::DoubleAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::DoubleAttribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::DoubleAttribute>();
    const auto& attr_str = ctx->String(std::to_string(attr_impl.data()));
    const auto& attr_data_val =
        ctx->Var("DataValue").Attr("float64").Call(attr_str);
    return ctx->Var("pir")
        .Attr(pir::DoubleAttribute::name())
        .Call(attr_data_val);
  }
};

template <>
struct AttrToAnfExprConverter<pir::Int32Attribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::Int32Attribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::Int32Attribute>();
    const auto& attr_str = ctx->String(std::to_string(attr_impl.data()));
    const auto& attr_data_val =
        ctx->Var("DataValue").Attr("int32").Call(attr_str);
    return ctx->Var("pir")
        .Attr(pir::Int32Attribute::name())
        .Call(attr_data_val);
  }
};

template <>
struct AttrToAnfExprConverter<pir::IndexAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::IndexAttribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::IndexAttribute>();
    const auto& attr_str = ctx->String(std::to_string(attr_impl.data()));
    const auto& attr_data_val =
        ctx->Var("DataValue").Attr("index").Call(attr_str);
    return ctx->Var("pir")
        .Attr(pir::IndexAttribute::name())
        .Call(attr_data_val);
  }
};

template <>
struct AttrToAnfExprConverter<pir::Int64Attribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::Int64Attribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::Int64Attribute>();
    const auto& attr_str = ctx->String(std::to_string(attr_impl.data()));
    const auto& attr_data_val =
        ctx->Var("DataValue").Attr("int64").Call(attr_str);
    return ctx->Var("pir")
        .Attr(pir::Int64Attribute::name())
        .Call(attr_data_val);
  }
};

template <>
struct AttrToAnfExprConverter<pir::PointerAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::PointerAttribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::PointerAttribute>();
    const auto& attr_str = ctx->String([&] {
      std::ostringstream ss;
      ss << attr_impl.data();
      return ss.str();
    }());
    const auto& attr_data_val =
        ctx->Var("PointerValue").Attr("void_ptr").Call(attr_str);
    return ctx->Var("pir")
        .Attr(pir::PointerAttribute::name())
        .Call(attr_data_val);
  }
};

template <>
struct AttrToAnfExprConverter<pir::TypeAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::TypeAttribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::TypeAttribute>();
    ADT_LET_CONST_REF(
        attr_data_val,
        PirToAnfExprHelper{}.ConvertPirTypeToAnfExpr(ctx, attr_impl.data()));
    return ctx->Var("pir").Attr(pir::TypeAttribute::name()).Call(attr_data_val);
  }
};

template <>
struct AttrToAnfExprConverter<pir::StrAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::StrAttribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::StrAttribute>();
    return ctx->Var("pir")
        .Attr(pir::StrAttribute::name())
        .Call(attr_impl.AsString());
  }
};

template <>
struct AttrToAnfExprConverter<pir::ArrayAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::ArrayAttribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::ArrayAttribute>();
    std::vector<axpr::AnfExpr> elts_anf_exprs{};
    const auto& data = attr_impl.AsVector();
    elts_anf_exprs.reserve(data.size());
    for (const auto& elt : data) {
      ADT_LET_CONST_REF(elt_anf_expr,
                        PirToAnfExprHelper{}.ConvertPirAttrToAnfExpr(ctx, elt));
      elts_anf_exprs.emplace_back(elt_anf_expr);
    }
    return ctx->Var("pir")
        .Attr(pir::ArrayAttribute::name())
        .Call(ctx->Var(axpr::kBuiltinList()).Apply(elts_anf_exprs));
  }
};

template <>
struct AttrToAnfExprConverter<pir::TensorNameAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<pir::TensorNameAttribute>());
    const auto& attr_impl = attr.template dyn_cast<pir::TensorNameAttribute>();
    return ctx->Var("pir")
        .Attr(pir::TensorNameAttribute::name())
        .Call(attr_impl.data());
  }
};

template <>
struct AttrToAnfExprConverter<pir::shape::SymbolAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    return adt::errors::NotImplementedError{
        std::string() + "pir." + pir::shape::SymbolAttribute::name() +
        "() not implemented"};
  }
};

template <>
struct AttrToAnfExprConverter<::paddle::dialect::KernelAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    return adt::errors::NotImplementedError{
        std::string() + "pir." + ::paddle::dialect::KernelAttribute::name() +
        "() not implemented"};
  }
};

template <>
struct AttrToAnfExprConverter<::paddle::dialect::IntArrayAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<::paddle::dialect::IntArrayAttribute>());
    const auto& attr_impl =
        attr.template dyn_cast<::paddle::dialect::IntArrayAttribute>();
    std::vector<axpr::AnfExpr> elts_anf_exprs{};
    const auto& phi_int_array = attr_impl.data();
    const auto& data = phi_int_array.GetData();
    elts_anf_exprs.reserve(data.size());
    for (const auto& elt : data) {
      const auto& elt_anf_expr = ctx->Int64(elt);
      elts_anf_exprs.emplace_back(elt_anf_expr);
    }
    return ctx->Var("pir")
        .Attr(::paddle::dialect::IntArrayAttribute::name())
        .Call(ctx->Var(axpr::kBuiltinList()).Apply(elts_anf_exprs));
  }
};

adt::Result<axpr::AnfExpr> ConvertToDataValueAnfExpr(
    axpr::LetContext* ctx, const phi::Scalar& scalar) {
  ADT_LET_CONST_REF(data_value, ScalarHelper{}.ConvertToDataValue(scalar));
  using RetT = adt::Result<axpr::AnfExpr>;
  return data_value.Match(
      [&](const axpr::complex64& impl) -> RetT {
        return ConvertToComplex64AnfExpr(ctx, impl);
      },
      [&](const axpr::complex128& impl) -> RetT {
        return ConvertToComplex128AnfExpr(ctx, impl);
      },
      [&](const auto& impl) -> RetT {
        try {
          const auto& data_type = data_value.GetType();
          const auto& data_val_str = ctx->String(scalar.ToRawString());
          return ctx->Var("DataValue")
              .Attr(data_type.Name())
              .Call(data_val_str);
        } catch (const std::exception& e) {
          return adt::errors::RuntimeError{e.what()};
        }
      });
}

template <>
struct AttrToAnfExprConverter<::paddle::dialect::ScalarAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<::paddle::dialect::ScalarAttribute>());
    const auto& attr_impl =
        attr.template dyn_cast<::paddle::dialect::ScalarAttribute>();
    ADT_LET_CONST_REF(data_value_anf_expr,
                      ConvertToDataValueAnfExpr(ctx, attr_impl.data()));
    return ctx->Var("pir")
        .Attr(::paddle::dialect::ScalarAttribute::name())
        .Call(data_value_anf_expr);
  }
};

template <>
struct AttrToAnfExprConverter<::paddle::dialect::DataTypeAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<::paddle::dialect::DataTypeAttribute>());
    const auto& attr_impl =
        attr.template dyn_cast<::paddle::dialect::DataTypeAttribute>();
    ADT_LET_CONST_REF(data_type,
                      axpr::GetDataTypeFromPhiDataType(attr_impl.data()));
    const auto& data_type_anf_expr =
        ctx->Var("DataType").Attr(data_type.Name());
    return ctx->Var("pir")
        .Attr(::paddle::dialect::DataTypeAttribute::name())
        .Call(data_type_anf_expr);
  }
};

adt::Result<axpr::AnfExpr> ConvertToPlaceAnfExpr(axpr::LetContext* ctx,
                                                 const phi::Place& place) {
  if (place.GetType() == phi::AllocationType::UNDEFINED) {
    return ctx->Var("pir").Attr("UndefinedPlace").Call();
  } else if (place.GetType() == phi::AllocationType::CPU) {
    return ctx->Var("pir").Attr("CPUPlace").Call();
  } else if (place.GetType() == phi::AllocationType::GPU) {
    const auto& device_id = ctx->Int64(place.GetDeviceId());
    return ctx->Var("pir").Attr("GPUPlace").Call(device_id);
  } else if (place.GetType() == phi::AllocationType::GPUPINNED) {
    return ctx->Var("pir").Attr("GPUPinnedPlace").Call();
  } else if (place.GetType() == phi::AllocationType::XPU) {
    const auto& device_id = ctx->Int64(place.GetDeviceId());
    return ctx->Var("pir").Attr("XPUPlace").Call(device_id);
  } else if (place.GetType() == phi::AllocationType::IPU) {
    const auto& device_id = ctx->Int64(place.GetDeviceId());
    return ctx->Var("pir").Attr("IPUPlace").Call(device_id);
  } else if (place.GetType() == phi::AllocationType::CUSTOM) {
    const auto& device_type = ctx->String(place.GetDeviceType());
    const auto& device_id = ctx->Int64(place.GetDeviceId());
    return ctx->Var("pir").Attr("CustomPlace").Call(device_type, device_id);
  }
  return adt::errors::TypeError{
      "ConvertToPlaceAnfExpr() failed. invalid place"};
}

template <>
struct AttrToAnfExprConverter<::paddle::dialect::PlaceAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<::paddle::dialect::PlaceAttribute>());
    const auto& attr_impl =
        attr.template dyn_cast<::paddle::dialect::PlaceAttribute>();
    ADT_LET_CONST_REF(place_anf_expr,
                      ConvertToPlaceAnfExpr(ctx, attr_impl.data()));
    return ctx->Var("pir")
        .Attr(::paddle::dialect::PlaceAttribute::name())
        .Call(place_anf_expr);
  }
};

template <>
struct AttrToAnfExprConverter<::paddle::dialect::DataLayoutAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    ADT_CHECK(attr.template isa<::paddle::dialect::DataLayoutAttribute>());
    const auto& attr_impl =
        attr.template dyn_cast<::paddle::dialect::DataLayoutAttribute>();
    ADT_LET_CONST_REF(data_layout_anf_expr,
                      ConvertToDataLayoutAnfExpr(ctx, attr_impl.data()));
    return ctx->Var("pir")
        .Attr(::paddle::dialect::DataLayoutAttribute::name())
        .Call(data_layout_anf_expr);
  }
};

template <>
struct AttrToAnfExprConverter<::cinn::dialect::GroupInfoAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    return adt::errors::NotImplementedError{
        std::string() + "pir." + ::cinn::dialect::GroupInfoAttribute::name() +
        "() is not implemneted"};
  }
};

template <>
struct AttrToAnfExprConverter<::cinn::dialect::CINNKernelInfoAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    return adt::errors::NotImplementedError{
        std::string() + "pir." +
        ::cinn::dialect::CINNKernelInfoAttribute::name() +
        "() is not implemneted"};
  }
};

template <>
struct AttrToAnfExprConverter<UnclassifiedAttribute> {
  static adt::Result<axpr::AnfExpr> Call(axpr::LetContext* ctx,
                                         pir::Attribute attr) {
    return adt::errors::NotImplementedError{std::string() + "pir." +
                                            UnclassifiedAttribute::name() +
                                            "() is not implemneted"};
  }
};

}  // namespace

adt::Result<axpr::AnfExpr> PirToAnfExprHelper::ConvertTypeToAnfExpr(
    axpr::LetContext* ctx, axpr::Value type) {
  ADT_LET_CONST_REF(pir_type, type.template CastTo<pir::Type>());
  return ConvertPirTypeToAnfExpr(ctx, pir_type);
}

adt::Result<axpr::AnfExpr> PirToAnfExprHelper::ConvertPirTypeToAnfExpr(
    axpr::LetContext* ctx, pir::Type type) {
  const auto& type_id = GetTypeAdtTypeId(type);
  using RetT = adt::Result<axpr::AnfExpr>;
  return type_id.Match([&](const auto& impl) -> RetT {
    using T = typename std::decay_t<decltype(impl)>::type;
    return TypeToAnfExprConverter<T>::Call(ctx, type);
  });
}

adt::Result<axpr::AnfExpr> PirToAnfExprHelper::ConvertAttrToAnfExpr(
    axpr::LetContext* ctx, axpr::Value attr) {
  ADT_LET_CONST_REF(pir_attr, attr.template CastTo<pir::Attribute>());
  return ConvertPirAttrToAnfExpr(ctx, pir_attr);
}

adt::Result<axpr::AnfExpr> PirToAnfExprHelper::ConvertPirAttrToAnfExpr(
    axpr::LetContext* ctx, pir::Attribute attr) {
  const auto& attr_id = GetAttrAdtTypeId(attr);
  using RetT = adt::Result<axpr::AnfExpr>;
  return attr_id.Match([&](const auto& impl) -> RetT {
    using T = typename std::decay_t<decltype(impl)>::type;
    return AttrToAnfExprConverter<T>::Call(ctx, attr);
  });
}

}  // namespace ap::paddle
