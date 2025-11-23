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

#include "paddle/ap/include/axpr/pointer_type_util.h"
#include "paddle/ap/include/code_gen/kernel_arg_id.h"
#include "paddle/ap/include/code_module/arg_type.h"

namespace ap::code_gen {

template <typename BirNode>
struct KernelArgIdHelper {
  using BirNativeIrValue = typename BirNode::native_value_type;

  template <typename ValueT>
  adt::Result<KernelArgId<BirNode>> CastToKernelArgId(const ValueT& val) {
    using RetT = adt::Result<KernelArgId<BirNode>>;
    return val.Match(
        [&](const DimExprKernelArgId<BirNode>& impl) -> RetT { return impl; },
        [&](const InTensorDataPtrKernelArgId<BirNode>& impl) -> RetT {
          return impl;
        },
        [&](const OutTensorDataPtrKernelArgId<BirNode>& impl) -> RetT {
          return impl;
        },
        [&](const auto& impl) -> RetT {
          return adt::errors::TypeError{
              std::string() +
              "only DimExprKernelArgId, InTensorDataPtrKernelArgId and "
              "OutTensorDataPtrKernelArgId (not including '" +
              axpr::GetTypeName(val) + "') can be cast to KernelArgId"};
        });
  }

  adt::Result<code_module::ArgType> GetArgType(
      const KernelArgId<BirNode>& arg_id) {
    using RetT = adt::Result<code_module::ArgType>;
    return arg_id.Match(
        [](const DimExprKernelArgId<BirNode>&) -> RetT {
          return axpr::CppDataType<int64_t>();
        },
        [&](const InTensorDataPtrKernelArgId<BirNode>& in_data_ptr) -> RetT {
          ADT_LET_CONST_REF(ir_value,
                            GetBirNativeIrValue(in_data_ptr->ir_value));
          ADT_LET_CONST_REF(data_type, ir_value.GetDataType());
          return axpr::GetConstPointerTypeFromDataType(data_type);
        },
        [&](const OutTensorDataPtrKernelArgId<BirNode>& out_data_ptr) -> RetT {
          ADT_LET_CONST_REF(ir_value,
                            GetBirNativeIrValue(out_data_ptr->ir_value));
          ADT_LET_CONST_REF(data_type, ir_value.GetDataType());
          return axpr::GetMutablePointerTypeFromDataType(data_type);
        });
  }

  adt::Result<BirNativeIrValue> GetBirNativeIrValue(
      const BirNode& bir_node) const {
    using RetT = adt::Result<BirNativeIrValue>;
    return bir_node.Match(
        [&](const BirNativeIrValue& impl) -> RetT { return impl; },
        [&](const typename BirNode::ref_value_type& impl) -> RetT {
          return impl.GetOwnerNativeIrValue();
        },
        [&](const auto& impl) -> RetT {
          using T = std::decay_t<decltype(impl)>;
          return adt::errors::NotImplementedError{
              std::string() +
              "GetBirNativeIrValue() failed. only 'NativeIrValue' and "
              "'RefIrValue' argument expected, but '" +
              typeid(T).name() + "' found."};
        });
  }
};

}  // namespace ap::code_gen
