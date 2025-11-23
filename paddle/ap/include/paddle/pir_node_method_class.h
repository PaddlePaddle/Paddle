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

#include "paddle/ap/include/axpr/dim_expr_method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/paddle/pir/attribute_method_class.h"
#include "paddle/ap/include/paddle/pir/shape_or_data_method_class.h"
#include "paddle/ap/include/paddle/pir/type_method_class.h"

namespace ap::paddle {

template <typename ValueT>
struct NativeIrValueMethodClass {
  using This = NativeIrValueMethodClass;
  using Self = NativeIrValue;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const auto* ptr = self.value.impl();
    ss << "<NativeIrValue object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<ValueT> Hash(const ValueT& self_val,
                                  const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return static_cast<int64_t>(std::hash<Self>()(self));
  }

  static adt::Result<ValueT> GetAttr(const ValueT& self_val,
                                     const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    const auto& attr_name_val = args.at(0);
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "dtype") {
      return This{}.GetDataType(self);
    }
    if (attr_name == "type") {
      return GetPirTypeClass().New(self.value.type());
    }
    return adt::errors::TypeError{std::string() +
                                  "NativeIrValue instance has no attribute '" +
                                  attr_name + "'."};
  }

  static adt::Result<ValueT> GetSymbolicShapeOrData(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 0);
    ADT_LET_CONST_REF(shape_or_data_ptr, self.GetShapeOrDataDimExprsPtr());
    return ap::paddle::GetPirShapeOrDataClass().New(*shape_or_data_ptr);
  }

  static adt::Result<ValueT> SymbolicShapeToList(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 0);
    return This{}.GetShape(self);
  }

  adt::Result<ValueT> GetShape(const Self& self) {
    ADT_LET_CONST_REF(shape_ptr, self.GetShapeDimExprsPtr());
    adt::List<ValueT> lst;
    lst->reserve(shape_ptr->size());
    for (const auto& dim_expr : *shape_ptr) {
      axpr::BuiltinClassInstance<ValueT> instance{
          axpr::GetDimExprClass<ValueT>(), dim_expr};
      lst->emplace_back(instance);
    }
    return lst;
  }

  adt::Result<ValueT> GetDataType(const Self& self) {
    ADT_LET_CONST_REF(dtype, self.GetDataType());
    return dtype;
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetNativeIrValueClass() {
  using ImplMethods = NativeIrValueMethodClass<ValueT>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("NativeIrValue", [&](const auto& Yield) {
        Yield("__getattr__", &ImplMethods::GetAttr);
        Yield("__str__", &ImplMethods::ToString);
        Yield("__hash__", &ImplMethods::Hash);
        Yield("symbolic_shape_to_list", &ImplMethods::SymbolicShapeToList);
        Yield("get_symbolic_shape_or_data",
              &ImplMethods::GetSymbolicShapeOrData);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename ImplMethods::Self>(cls);
}

template <typename ValueT>
struct PackedIrValueMethodClass {
  using This = PackedIrValueMethodClass;
  using Self = PackedIrValue;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const pir::Operation* ptr = self.fusion_op;
    std::ostringstream ss;
    ss << "<PackedIrValue object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<ValueT> Hash(const ValueT& self_val,
                                  const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const pir::Operation* ptr = self.fusion_op;
    return reinterpret_cast<int64_t>(ptr);
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetPackedIrValueClass() {
  using ImplMethods = PackedIrValueMethodClass<ValueT>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("PackedIrValue", [&](const auto& Yield) {
        Yield("__str__", &ImplMethods::ToString);
        Yield("__hash__", &ImplMethods::Hash);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename ImplMethods::Self>(cls);
}

template <typename ValueT>
struct RefIrValueMethodClass {
  using This = RefIrValueMethodClass;
  using Self = RefIrValue;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const auto* ptr = self.ref_node_info.__adt_rc_shared_ptr_raw_ptr();
    ss << "<RefIrValue object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<ValueT> Hash(const ValueT& self_val,
                                  const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return reinterpret_cast<int64_t>(
        self.ref_node_info.__adt_rc_shared_ptr_raw_ptr());
  }

  static adt::Result<ValueT> GetAttr(const ValueT& self_val,
                                     const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    const auto& attr_name_val = args.at(0);
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "dtype") {
      return This{}.GetDataType(self);
    }
    return adt::errors::TypeError{std::string() +
                                  "NativeIrValue instance has no attribute '" +
                                  attr_name + "'."};
  }

  static adt::Result<ValueT> SymbolicShapeToList(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 0);
    return This{}.GetShape(self);
  }

  adt::Result<ValueT> GetShape(const Self& self) {
    ADT_LET_CONST_REF(ir_value, self.GetOwnerNativeIrValue());
    ADT_LET_CONST_REF(shape_ptr, ir_value.GetShapeDimExprsPtr());
    adt::List<ValueT> lst;
    lst->reserve(shape_ptr->size());
    for (const auto& dim_expr : *shape_ptr) {
      axpr::BuiltinClassInstance<ValueT> instance{
          axpr::GetDimExprClass<ValueT>(), dim_expr};
      lst->emplace_back(instance);
    }
    return lst;
  }

  adt::Result<ValueT> GetDataType(const Self& self) {
    ADT_LET_CONST_REF(ir_value, self.GetOwnerNativeIrValue());
    ADT_LET_CONST_REF(dtype, ir_value.GetDataType());
    return dtype;
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetRefIrValueClass() {
  using ImplMethods = RefIrValueMethodClass<ValueT>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("RefIrValue", [&](const auto& Yield) {
        Yield("__getattr__", &ImplMethods::GetAttr);
        Yield("__str__", &ImplMethods::ToString);
        Yield("__hash__", &ImplMethods::Hash);
        Yield("symbolic_shape_to_list", &ImplMethods::SymbolicShapeToList);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename ImplMethods::Self>(cls);
}

template <typename ValueT>
struct NativeIrOpMethodClass {
  using This = NativeIrOpMethodClass;
  using Self = NativeIrOp;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const auto* ptr = self.op;
    ss << "<NativeIrOp object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<ValueT> Hash(const ValueT& self_val,
                                  const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const pir::Operation* ptr = self.op;
    return reinterpret_cast<int64_t>(ptr);
  }

  static adt::Result<ValueT> GetAttr(const ValueT& self_val,
                                     const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(attr_name, args.at(0).template CastTo<std::string>());
    const pir::Operation* ptr = self.op;
    const auto& attrs = ptr->attributes();
    const auto& iter = attrs.find(attr_name);
    if (iter == attrs.end()) {
      return adt::errors::KeyError{
          "NativeIrOp.__getattr__() failed. can not found attribute '" +
          attr_name + "'"};
    }
    return GetPirAttributeClass().New(iter->second);
  }

  static adt::Result<ValueT> NumOperands(const ValueT& self_val,
                                         const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 0) << adt::errors::TypeError{
        std::string() + "NativeIrOp.num_operands() takes 0 arguments, but " +
        std::to_string(args.size()) + " were given"};
    const pir::Operation* op = self.op;
    int64_t num_operands = op->num_operands();
    return num_operands;
  }

  static adt::Result<ValueT> OperandSource(const ValueT& self_val,
                                           const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "NativeIrOp.operand_source() takes 1 argument, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(i, args.at(0).template CastTo<int64_t>());
    const pir::Operation* op = self.op;
    ADT_CHECK(i >= 0);
    ADT_CHECK(i < op->num_operands());
    pir::Value value = op->operand_source(i);
    return GetNativeIrValueClass<axpr::Value>().New(NativeIrValue{value});
  }

  static adt::Result<ValueT> NumResults(const ValueT& self_val,
                                        const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 0) << adt::errors::TypeError{
        std::string() + "NativeIrOp.num_results() takes 0 arguments, but " +
        std::to_string(args.size()) + " were given"};
    const pir::Operation* op = self.op;
    int64_t num_results = op->num_results();
    return num_results;
  }

  static adt::Result<ValueT> Result(const ValueT& self_val,
                                    const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "NativeIrOp.result() takes 1 argument, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(i, args.at(0).template CastTo<int64_t>());
    const pir::Operation* op = self.op;
    ADT_CHECK(i >= 0);
    ADT_CHECK(i < op->num_results());
    pir::Value value = op->result(i);
    return GetNativeIrValueClass<axpr::Value>().New(NativeIrValue{value});
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetNativeIrOpClass() {
  using Impl = NativeIrOpMethodClass<ValueT>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("NativeIrOp", [&](const auto& Yield) {
        Yield("__str__", &Impl::ToString);
        Yield("__hash__", &Impl::Hash);
        Yield("__getattr__", &Impl::GetAttr);
        Yield("num_operands", &Impl::NumOperands);
        Yield("operand_source", &Impl::OperandSource);
        Yield("num_results", &Impl::NumResults);
        Yield("result", &Impl::Result);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

template <typename ValueT>
struct PackedIrOpMethodClass {
  using This = PackedIrOpMethodClass;
  using Self = PackedIrOp;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const pir::Operation* ptr = self.fusion_op;
    ss << "<PackedIrOp object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<ValueT> Hash(const ValueT& self_val,
                                  const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const pir::Operation* ptr = self.fusion_op;
    return reinterpret_cast<int64_t>(ptr);
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetPackedIrOpClass() {
  using Impl = PackedIrOpMethodClass<ValueT>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("PackedIrOp", [&](const auto& Yield) {
        Yield("__str__", &Impl::ToString);
        Yield("__hash__", &Impl::Hash);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

template <typename ValueT>
struct RefIrOpMethodClass {
  using This = RefIrOpMethodClass;
  using Self = RefIrOp;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const auto* ptr = self.ref_node_info.__adt_rc_shared_ptr_raw_ptr();
    ss << "<RefIrOp object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<ValueT> Hash(const ValueT& self_val,
                                  const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return reinterpret_cast<int64_t>(
        self.ref_node_info.__adt_rc_shared_ptr_raw_ptr());
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetRefIrOpClass() {
  using Impl = RefIrOpMethodClass<ValueT>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("RefIrOp", [&](const auto& Yield) {
        Yield("__str__", &Impl::ToString);
        Yield("__hash__", &Impl::Hash);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

}  // namespace ap::paddle
