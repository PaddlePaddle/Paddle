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
#include "paddle/ap/include/kernel_dispatch/mutable_tensor.h"

namespace ap::kernel_dispatch {

using ap::axpr::BuiltinBinaryFunc;
using ap::axpr::BuiltinFuncType;
using ap::axpr::BuiltinUnaryFunc;
using ap::axpr::CppDataType;
using ap::axpr::CppPointerType;
using ap::axpr::DataType;
using ap::axpr::DataValue;
using ap::axpr::Method;
using ap::axpr::MethodClass;
using ap::axpr::PointerType;
using ap::axpr::PointerValue;

namespace detail {

template <typename Val>
Result<Val> MutableTensorShapeGetAttr(const MutableTensor<Val>& tensor,
                                      const std::string&) {
  return tensor->dims;
}

template <typename T>
T* GetMutableTensorDataPtr(const ap::axpr::CppDataType<T>&,
                           const MutableTensorData& tensor) {
  return tensor.template data<T>();
}

template <typename Val>
Result<Val> MutableTensorDataGetAttr(const MutableTensor<Val>& tensor,
                                     const std::string&) {
  phi::DataType dtype = tensor->tensor_data.dtype();
  const auto& data_type = ap::axpr::GetDataTypeFromPhiDataType(dtype);
  ADT_RETURN_IF_ERR(data_type);
  return data_type.GetOkValue().Match(
      [&](const adt::Undefined&) -> Result<Val> {
        return TypeError{"dtype is invalid."};
      },
      [&](const auto& impl) -> Result<Val> {
        return PointerValue{GetMutableTensorDataPtr(impl, tensor->tensor_data)};
      });
}

template <typename Val>
using MutableTensorGetAttrT = Result<Val> (*)(const MutableTensor<Val>& tensor,
                                              const std::string&);

template <typename Val>
Result<Val> TensorGetAttr(const MutableTensor<Val>& tensor,
                          const std::string& name) {
  static const std::unordered_map<std::string, MutableTensorGetAttrT<Val>> map{
      {"shape", &MutableTensorShapeGetAttr<Val>},
      {"data_ptr", &MutableTensorDataGetAttr<Val>},
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
struct MutableTensorMethodClass {
  using Self = MutableTensor<ValueT>;

  static adt::Result<ValueT> GetAttr(const ValueT& obj_val,
                                     const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1);
    const auto& attr_name_val = args.at(0);
    ADT_LET_CONST_REF(obj, axpr::Get<MutableTensor<ValueT>>(obj_val));
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    return detail::TensorGetAttr<ValueT>(obj, attr_name);
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetMutableTensorClass() {
  using Methods = MutableTensorMethodClass<ValueT>;
  static auto cls(axpr::MakeBuiltinClass<ValueT>(
      "MutableTensor",
      [&](const auto& Yield) { Yield("__getattr__", &Methods::GetAttr); }));
  using Self = typename Methods::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::kernel_dispatch
