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

#include "paddle/ap/include/paddle/pir/shape_or_data_method_class.h"
#include "paddle/ap/include/axpr/abstract_list.h"
#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/dim_expr.h"
#include "paddle/ap/include/paddle/pir/type_adt_type_id.h"
#include "paddle/ap/include/paddle/pir/type_method_class.h"

namespace ap::paddle {

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetPirShapeOrDataClass();

adt::Result<axpr::Value> PirShapeOrDataString(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(self,
                    self_val.template CastTo<symbol::ShapeOrDataDimExprs>());
  std::ostringstream ss;
  ss << self;
  return ss.str();
}

adt::Result<adt::List<axpr::Value>> GetConstructorArgsImpl(
    const symbol::NullShapeOrDataDimExpr& impl) {
  return adt::List<axpr::Value>{};
}

adt::Result<adt::List<axpr::Value>> GetConstructorArgsImpl(
    const symbol::TensorShapeOrDataDimExprs& impl) {
  adt::List<axpr::Value> shape{};
  shape->reserve(impl.shape().size());
  for (const auto& dim_expr : impl.shape()) {
    shape->emplace_back(axpr::GetDimExprClass<axpr::Value>().New(dim_expr));
  }
  if (impl.data().has_value()) {
    adt::List<axpr::Value> data{};
    data->reserve(impl.data().value().size());
    for (const auto& dim_expr : impl.data().value()) {
      data->emplace_back(axpr::GetDimExprClass<axpr::Value>().New(dim_expr));
    }
    return adt::List<axpr::Value>{axpr::Value{shape}, axpr::Value{data}};
  } else {
    return adt::List<axpr::Value>{axpr::Value{shape},
                                  axpr::Value{adt::Nothing{}}};
  }
}

adt::Result<adt::List<axpr::Value>> GetConstructorArgsImpl(
    const symbol::TensorListShapeOrDataDimExprs& impl) {
  adt::List<axpr::Value> lst{};
  lst->reserve(impl.size());
  for (const auto& shape_or_data : impl) {
    lst->push_back(GetPirShapeOrDataClass().New(shape_or_data));
  }
  return adt::List<axpr::Value>{axpr::Value{lst}};
}

adt::Result<adt::List<axpr::Value>> GetConstructorArgsImpl(
    const symbol::RankedTensorArrayShapeOrDataDimExprs& impl) {
  return adt::errors::NotImplementedError{
      "pir.s_ranked_tensor_array_shape_or_data not implemented"};
}

std::string PirShapeOrDataGetTypeNameImpl(
    const symbol::ShapeOrDataDimExprs& self) {
  return self.Match(
      [](const symbol::NullShapeOrDataDimExpr&) -> std::string {
        return "s_null";
      },
      [](const symbol::TensorShapeOrDataDimExprs&) -> std::string {
        return "s_tensor_shape_or_data";
      },
      [](const symbol::TensorListShapeOrDataDimExprs&) -> std::string {
        return "s_tensor_list_shape_or_data";
      },
      [](const symbol::RankedTensorArrayShapeOrDataDimExprs&) -> std::string {
        return "s_ranked_tensor_array_shape_or_data";
      });
}

adt::Result<axpr::Value> PirShapeOrDataGetTypeName(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  ADT_LET_CONST_REF(self,
                    self_val.template CastTo<symbol::ShapeOrDataDimExprs>());
  return PirShapeOrDataGetTypeNameImpl(self);
}

adt::Result<axpr::Value> PirShapeOrDataMatch(
    axpr::InterpreterBase<axpr::Value>* interpreter,
    const axpr::Value& self_val,
    const std::vector<axpr::Value>& packed_args_val) {
  ADT_LET_CONST_REF(self,
                    self_val.template CastTo<symbol::ShapeOrDataDimExprs>());
  const auto& packed_args =
      axpr::CastToPackedArgs<axpr::Value>(packed_args_val);
  const auto& type_name = PirShapeOrDataGetTypeNameImpl(self);
  const auto& [args, kwargs] = *packed_args;
  ADT_CHECK(args->size() == 0) << adt::errors::TypeError{
      std::string() +
      "PirShapeOrData.match() supports keyword arguments only, but " +
      std::to_string(args->size()) + " positional arguments were given"};

  std::string key = type_name;
  if (!kwargs->Has(type_name)) {
    if (!kwargs->Has("_")) {
      return adt::errors::TypeError{
          std::string() + "PirShapeOrData.match() failed. no keyword '" +
          type_name + "' or '_' provided"};
    }
    key = "_";
  }
  ADT_LET_CONST_REF(func, kwargs->Get(key));
  auto GetConstructorArgs =
      [&](const auto& impl) -> adt::Result<adt::List<axpr::Value>> {
    return GetConstructorArgsImpl(impl);
  };
  ADT_LET_CONST_REF(shape_or_data_constructor_args,
                    self.Match(GetConstructorArgs));
  ADT_CHECK(axpr::CallableHelper{}.IsCallable(func)) << adt::errors::TypeError{
      std::string() +
      "the arguments of PirShapeOrData.match() should be callable"};
  if (key == "_") {
    return interpreter->InterpretCall(func, {});
  } else {
    return interpreter->InterpretCall(func,
                                      shape_or_data_constructor_args.vector());
  }
}

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetPirShapeOrDataClass() {
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "PirShapeOrData", [&](const auto& Yield) {
        Yield("__str__", &PirShapeOrDataString);
        Yield("get_type_name", &PirShapeOrDataGetTypeName);
        Yield("match", &PirShapeOrDataMatch);
      }));
  return axpr::MakeGlobalNaiveClassOps<symbol::ShapeOrDataDimExprs>(cls);
}

adt::Result<axpr::Value> MakeNullShapeOrDataDimExpr(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  symbol::ShapeOrDataDimExprs shape_or_data{symbol::NullShapeOrDataDimExpr{}};
  return GetPirShapeOrDataClass().New(shape_or_data);
}

adt::Result<std::vector<symbol::DimExpr>> GetDimExprs(
    const axpr::Value& dim_exprs) {
  ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(dim_exprs));
  std::vector<symbol::DimExpr> ret;
  ADT_LET_CONST_REF(lst_size, lst.size());
  ret.reserve(lst_size);
  for (int i = 0; i < lst_size; ++i) {
    ADT_LET_CONST_REF(elt_val, lst.at(i));
    ADT_LET_CONST_REF(elt, elt_val.template CastTo<symbol::DimExpr>());
    ret.emplace_back(elt);
  }
  return ret;
}

adt::Result<std::optional<std::vector<symbol::DimExpr>>> GetDataByArgVec(
    const std::vector<axpr::Value>& args) {
  if (args.size() != 2) {
    return std::nullopt;
  }
  if (args.at(1).template CastableTo<adt::Nothing>()) {
    return std::nullopt;
  }
  ADT_LET_CONST_REF(dim_exprs, GetDimExprs(args.at(1)));
  return dim_exprs;
}

adt::Result<axpr::Value> MakeTensorShapeOrDataDimExprs(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1 || args.size() == 2);
  ADT_LET_CONST_REF(shape, GetDimExprs(args.at(0)));
  ADT_LET_CONST_REF(opt_data, GetDataByArgVec(args));
  if (opt_data.has_value()) {
    symbol::TensorShapeOrDataDimExprs tensor_shape_or_data{shape,
                                                           opt_data.value()};
    symbol::ShapeOrDataDimExprs shape_or_data{tensor_shape_or_data};
    return GetPirShapeOrDataClass().New(shape_or_data);
  } else {
    symbol::TensorShapeOrDataDimExprs tensor_shape_or_data{shape};
    symbol::ShapeOrDataDimExprs shape_or_data{tensor_shape_or_data};
    return GetPirShapeOrDataClass().New(shape_or_data);
  }
}

adt::Result<axpr::Value> MakeTensorListShapeOrDataDimExprs(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(lst, axpr::AbstractList<axpr::Value>::CastFrom(args.at(0)));
  std::vector<symbol::TensorShapeOrDataDimExprs> elts;
  ADT_LET_CONST_REF(lst_size, lst.size());
  elts.reserve(lst_size);
  for (int i = 0; i < lst_size; ++i) {
    ADT_LET_CONST_REF(elt_val, lst.at(i));
    ADT_LET_CONST_REF(
        elt, elt_val.template CastTo<symbol::TensorShapeOrDataDimExprs>());
    elts.emplace_back(elt);
  }
  symbol::ShapeOrDataDimExprs shape_or_data{elts};
  return GetPirShapeOrDataClass().New(shape_or_data);
}

adt::Result<axpr::Value> MakeRankedTensorArrayShapeOrDataDimExprs(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  return adt::errors::NotImplementedError{
      "pir.s_ranked_tensor_array_shape_or_data not implemented"};
}

}  // namespace ap::paddle
