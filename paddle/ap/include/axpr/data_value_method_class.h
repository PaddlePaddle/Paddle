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

#include "paddle/ap/include/axpr/builtin_func_type.h"
#include "paddle/ap/include/axpr/constants.h"
#include "paddle/ap/include/axpr/data_value.h"
#include "paddle/ap/include/axpr/data_value_util.h"
#include "paddle/ap/include/axpr/method.h"
#include "paddle/ap/include/axpr/method_class.h"

namespace ap::axpr {

namespace detail {

template <typename Val>
Result<Val> ArgValueStaticCast(const Val& self, const std::vector<Val>& args) {
  if (args.size() != 1) {
    return TypeError{std::string() + "'DataValue.cast' take 1 arguments. but " +
                     std::to_string(args.size()) + " were given."};
  }
  const Result<DataValue>& arg_value = self.template TryGet<DataValue>();
  ADT_RETURN_IF_ERR(arg_value);
  const Result<DataType>& arg_type = args.at(0).template TryGet<DataType>();
  ADT_RETURN_IF_ERR(arg_type);
  const auto& data_value =
      arg_value.GetOkValue().StaticCastTo(arg_type.GetOkValue());
  ADT_RETURN_IF_ERR(data_value);
  return data_value.GetOkValue();
}

template <typename ValueT>
adt::Result<ValueT> DataValueGetAttr(const DataValue& data_val,
                                     const std::string& attr_name) {
  if (attr_name == "cast") {
    return ap::axpr::Method<ValueT>{data_val, &ArgValueStaticCast<ValueT>};
  }
  return adt::errors::AttributeError{"'DataValue' object has no attribute '" +
                                     attr_name + "'"};
}

}  // namespace detail

template <typename ValueT>
struct DataValueMethodClass {
  using This = DataValueMethodClass;
  using Self = DataValue;

  adt::Result<ValueT> ToString(const Self& self) {
    ADT_LET_CONST_REF(str, self.ToString());
    return str;
  }

  adt::Result<ValueT> Hash(const Self& self) {
    ADT_LET_CONST_REF(hash_value, self.GetHashValue());
    return hash_value;
  }

  template <typename BuiltinUnarySymbol>
  static BuiltinUnaryFunc<ValueT> GetBuiltinUnaryFunc() {
    if constexpr (ConvertBuiltinSymbolToArithmetic<
                      BuiltinUnarySymbol>::convertible) {
      using ArithmeticOp = typename ConvertBuiltinSymbolToArithmetic<
          BuiltinUnarySymbol>::arithmetic_op_type;
      return &This::UnaryFunc<ArithmeticOp>;
    } else {
      return adt::Nothing{};
    }
  }

  template <typename BuiltinBinarySymbol>
  static BuiltinBinaryFunc<ValueT> GetBuiltinBinaryFunc() {
    if constexpr (ConvertBuiltinSymbolToArithmetic<
                      BuiltinBinarySymbol>::convertible) {
      using ArithmeticOp = typename ConvertBuiltinSymbolToArithmetic<
          BuiltinBinarySymbol>::arithmetic_op_type;
      return &This::template BinaryFunc<ArithmeticOp>;
    } else if constexpr (std::is_same_v<BuiltinBinarySymbol,  // NOLINT
                                        builtin_symbol::GetAttr>) {
      return &This::GetAttr;
    } else {
      return adt::Nothing{};
    }
  }

  static adt::Result<ValueT> GetAttr(const ValueT& obj_val,
                                     const ValueT& attr_name_val) {
    const auto& opt_obj = obj_val.template TryGet<DataValue>();
    ADT_RETURN_IF_ERR(opt_obj);
    const auto& obj = opt_obj.GetOkValue();
    const auto& opt_attr_name = attr_name_val.template TryGet<std::string>();
    ADT_RETURN_IF_ERR(opt_attr_name);
    const auto& attr_name = opt_attr_name.GetOkValue();
    return detail::DataValueGetAttr<ValueT>(obj, attr_name);
  }

  template <typename ArithmeticOp>
  static adt::Result<ValueT> BinaryFunc(const ValueT& lhs_val,
                                        const ValueT& rhs_val) {
    const auto& opt_lhs = lhs_val.template TryGet<DataValue>();
    ADT_RETURN_IF_ERR(opt_lhs);
    const auto& lhs = opt_lhs.GetOkValue();
    const auto& opt_rhs = rhs_val.template TryGet<DataValue>();
    ADT_RETURN_IF_ERR(opt_rhs);
    const auto& rhs = opt_rhs.GetOkValue();
    const auto& ret = ArithmeticBinaryFunc<ArithmeticOp>(lhs, rhs);
    ADT_RETURN_IF_ERR(ret);
    return ret.GetOkValue();
  }

  template <typename ArithmeticOp>
  static adt::Result<ValueT> UnaryFunc(const ValueT& val) {
    const auto& opt_operand = val.template TryGet<DataValue>();
    ADT_RETURN_IF_ERR(opt_operand);
    const auto& operand = opt_operand.GetOkValue();
    const auto& ret = ArithmeticUnaryFunc<ArithmeticOp>(operand);
    ADT_RETURN_IF_ERR(ret);
    return ret.GetOkValue();
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, DataValue>
    : public DataValueMethodClass<ValueT> {};

namespace detail {

template <typename ValueT>
adt::Result<ValueT> ConstructDataValue(const ValueT&,
                                       const std::vector<ValueT>& args) {
  if (args.size() != 1) {
    return adt::errors::TypeError{
        std::string() + "constructor of 'DataValue' takes 1 arguments, but " +
        std::to_string(args.size()) + " were given."};
  }
  return args.at(0).Match(
      [](bool c) -> adt::Result<ValueT> { return DataValue{c}; },
      [](int64_t c) -> adt::Result<ValueT> { return DataValue{c}; },
      [](const DataValue& c) -> adt::Result<ValueT> { return c; },
      [&](const auto& impl) -> adt::Result<ValueT> {
        return adt::errors::TypeError{
            std::string() +
            "unsupported operand type for constructor of 'DataValue': '" +
            axpr::GetTypeName(args.at(0)) + "'"};
      });
}

}  // namespace detail

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<DataValue>> {
  using This = MethodClassImpl<ValueT, TypeImpl<DataValue>>;
  using Self = TypeImpl<DataValue>;

  adt::Result<ValueT> Call(const Self& self_val) {
    return &detail::ConstructDataValue<ValueT>;
  }

  adt::Result<ValueT> GetAttr(const Self&, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template CastTo<std::string>());
    static const std::map<std::string, axpr::BuiltinFuncType<ValueT>> map{
        {"float32", &This::MakeFloat32},
        {"float64", &This::MakeFloat64},
        // {"float16", &This::Make<float16>},
        // {"bfloat16", &This::Make<bfloat16>},
        {"int64", &This::MakeInt64},
        {"int64_t", &This::MakeInt64},
        {"int32", &This::MakeInt32},
        {"int32_t", &This::MakeInt32},
        {"int16", &This::MakeInt16},
        {"int16_t", &This::MakeInt16},
        {"int8", &This::MakeInt8},
        {"int8_t", &This::MakeInt8},
        {"uint64", &This::MakeUint64},
        {"uint64_t", &This::MakeUint64},
        {"uint32", &This::MakeUint32},
        {"uint32_t", &This::MakeUint32},
        {"uint16", &This::MakeUint16},
        {"uint16_t", &This::MakeUint16},
        {"uint8", &This::MakeUint8},
        {"uint8_t", &This::MakeUint8},
        {"bool", &This::MakeBool},
        {"complex64", &This::MakeComplex64},
        {"complex128", &This::MakeComplex128}};
    const auto& iter = map.find(attr_name);
    if (iter != map.end()) {
      return ValueT{iter->second};
    }
    return adt::errors::NotImplementedError{std::string() + "DataValue." +
                                            attr_name + "() not implemented"};
  }

  template <typename T>
  static T StrToNum(const std::string& str) {
    T x;
    std::stringstream ss;
    ss << str;
    ss >> x;
    return x;
  }

  static adt::Result<ValueT> MakeFloat32(const ValueT&,
                                         const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "DataValue.float32() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(str, args.at(0).template CastTo<std::string>());
    return DataValue{StrToNum<float>(str)};
  }

  static adt::Result<ValueT> MakeFloat64(const ValueT&,
                                         const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "DataValue.float64() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(str, args.at(0).template CastTo<std::string>());
    return DataValue{StrToNum<double>(str)};
  }

  static adt::Result<ValueT> MakeInt64(const ValueT&,
                                       const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "DataValue.int64() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(str, args.at(0).template CastTo<std::string>());
    return DataValue{StrToNum<int64_t>(str)};
  }

  static adt::Result<ValueT> MakeInt32(const ValueT&,
                                       const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "DataValue.int32() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(str, args.at(0).template CastTo<std::string>());
    return DataValue{StrToNum<int32_t>(str)};
  }

  static adt::Result<ValueT> MakeInt16(const ValueT&,
                                       const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "DataValue.int16() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(str, args.at(0).template CastTo<std::string>());
    return DataValue{StrToNum<int16_t>(str)};
  }

  static adt::Result<ValueT> MakeInt8(const ValueT&,
                                      const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "DataValue.int8() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(str, args.at(0).template CastTo<std::string>());
    return DataValue{StrToNum<int8_t>(str)};
  }

  static adt::Result<ValueT> MakeUint64(const ValueT&,
                                        const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "DataValue.uint64() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(str, args.at(0).template CastTo<std::string>());
    return DataValue{StrToNum<uint64_t>(str)};
  }

  static adt::Result<ValueT> MakeUint32(const ValueT&,
                                        const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "DataValue.uint32() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(str, args.at(0).template CastTo<std::string>());
    return DataValue{StrToNum<uint32_t>(str)};
  }

  static adt::Result<ValueT> MakeUint16(const ValueT&,
                                        const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "DataValue.uint16() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(str, args.at(0).template CastTo<std::string>());
    return DataValue{StrToNum<uint16_t>(str)};
  }

  static adt::Result<ValueT> MakeUint8(const ValueT&,
                                       const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "DataValue.uint8() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(str, args.at(0).template CastTo<std::string>());
    return DataValue{StrToNum<uint8_t>(str)};
  }

  static adt::Result<ValueT> MakeBool(const ValueT&,
                                      const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "DataValue.bool() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(str, args.at(0).template CastTo<std::string>());
    return DataValue{StrToNum<bool>(str)};
  }

  static adt::Result<ValueT> MakeComplex64(const ValueT&,
                                           const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() + "DataValue.complex64() takes 2 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(real_val, args.at(0).template CastTo<axpr::DataValue>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of DataValue.complex64() should be a DataValue, "
               "but a " +
               axpr::GetTypeName(args.at(0)) + " were given"};
    ADT_LET_CONST_REF(real, real_val.template TryGet<float>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of DataValue.complex64() should be a float32, "
               "but a " +
               real_val.GetType().Name() + " were given"};
    ADT_LET_CONST_REF(imag_val, args.at(1).template CastTo<axpr::DataValue>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 2 of DataValue.complex64() should be a DataValue, "
               "but a " +
               axpr::GetTypeName(args.at(1)) + " were given"};
    ADT_LET_CONST_REF(imag, real_val.template TryGet<float>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 2 of DataValue.complex64() should be a float32, "
               "but a " +
               imag_val.GetType().Name() + " were given"};
    return DataValue{axpr::complex64(real, imag)};
  }

  static adt::Result<ValueT> MakeComplex128(const ValueT&,
                                            const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() + "DataValue.complex128() takes 2 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(real_val, args.at(0).template CastTo<axpr::DataValue>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of DataValue.complex128() should be a "
               "DataValue, but a " +
               axpr::GetTypeName(args.at(0)) + " were given"};
    ADT_LET_CONST_REF(real, real_val.template TryGet<double>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of DataValue.complex128() should be a float64, "
               "but a " +
               real_val.GetType().Name() + " were given"};
    ADT_LET_CONST_REF(imag_val, args.at(1).template CastTo<axpr::DataValue>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 2 of DataValue.complex128() should be a "
               "DataValue, but a " +
               axpr::GetTypeName(args.at(1)) + " were given"};
    ADT_LET_CONST_REF(imag, real_val.template TryGet<double>())
        << adt::errors::TypeError{
               std::string() +
               "the argument 2 of DataValue.complex128() should be a float64, "
               "but a " +
               imag_val.GetType().Name() + " were given"};
    return DataValue{axpr::complex128(real, imag)};
  }
};

}  // namespace ap::axpr
