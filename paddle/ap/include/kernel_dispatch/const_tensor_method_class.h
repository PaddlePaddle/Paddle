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

#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/kernel_dispatch/const_tensor.h"

namespace ap::kernel_dispatch {

using ap::axpr::BuiltinBinaryFunc;
using ap::axpr::BuiltinFuncType;
using ap::axpr::BuiltinUnaryFunc;
using ap::axpr::CppDataType;
using ap::axpr::CppPointerType;
using ap::axpr::DataType;
using ap::axpr::Method;
using ap::axpr::MethodClass;
using ap::axpr::PointerType;
using ap::axpr::PointerValue;

namespace detail {

template <typename Val>
Result<Val> ConstTensorShapeGetAttr(const ConstTensor<Val>& tensor,
                                    const std::string&) {
  return tensor->dims;
}

template <typename T>
const T* GetConstTensorDataPtr(const ap::axpr::CppDataType<T>&,
                               const ConstTensorData& tensor) {
  return tensor.template data<T>();
}

template <typename Val>
Result<Val> ConstTensorDataGetAttr(const ConstTensor<Val>& tensor,
                                   const std::string&) {
  phi::DataType dtype = tensor->tensor_data.dtype();
  const auto& data_type = ap::axpr::GetDataTypeFromPhiDataType(dtype);
  ADT_RETURN_IF_ERR(data_type);
  return data_type.GetOkValue().Match(
      [&](const adt::Undefined&) -> Result<Val> {
        return TypeError{"dtype is invalid."};
      },
      [&](const auto& impl) -> Result<Val> {
        return PointerValue{GetConstTensorDataPtr(impl, tensor->tensor_data)};
      });
}

template <typename Val>
using ConstTensorGetAttrT = Result<Val> (*)(const ConstTensor<Val>& tensor,
                                            const std::string&);

template <typename Val>
Result<Val> TensorGetAttr(const ConstTensor<Val>& tensor,
                          const std::string& name) {
  static const std::unordered_map<std::string, ConstTensorGetAttrT<Val>> map{
      {"shape", &ConstTensorShapeGetAttr<Val>},
      {"data_ptr", &ConstTensorDataGetAttr<Val>},
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return AttributeError{std::string("'Tensor' has no attribute '") + name +
                          "'"};
  }
  return iter->second(tensor, name);
}

}  // namespace detail

template <typename ValueT>
struct ConstTensorMethodClass {
  using Self = ConstTensor<ValueT>;

  static adt::Result<ValueT> GetAttr(const ValueT& obj_val,
                                     const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1);
    const auto& attr_name_val = args.at(0);
    ADT_LET_CONST_REF(obj, axpr::Get<ConstTensor<ValueT>>(obj_val));
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    return detail::TensorGetAttr<ValueT>(obj, attr_name);
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetConstTensorClass() {
  using ImplMethods = ConstTensorMethodClass<ValueT>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("ConstTensor", [&](const auto& DoEach) {
        DoEach("__getattr__", &ImplMethods::GetAttr);
      }));
  using Self = typename ImplMethods::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::kernel_dispatch
