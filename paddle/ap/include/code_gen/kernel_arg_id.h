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
#include "paddle/ap/include/code_gen/dim_expr_kernel_arg_id.h"
#include "paddle/ap/include/code_gen/in_tensor_data_ptr_kernel_arg_id.h"
#include "paddle/ap/include/code_gen/out_tensor_data_ptr_kernel_arg_id.h"

namespace ap::code_gen {

template <typename BirNode>
using KernelArgIdImpl = std::variant<DimExprKernelArgId<BirNode>,
                                     InTensorDataPtrKernelArgId<BirNode>,
                                     OutTensorDataPtrKernelArgId<BirNode>>;

template <typename BirNode>
struct KernelArgId : public KernelArgIdImpl<BirNode> {
  using KernelArgIdImpl<BirNode>::KernelArgIdImpl;

  ADT_DEFINE_VARIANT_METHODS(KernelArgIdImpl<BirNode>);

  template <typename ValueT>
  ValueT CastTo() const {
    return Match([](const auto& impl) -> ValueT { return impl; });
  }

  template <typename ValueT>
  static adt::Result<KernelArgId> CastFrom(const ValueT& val) {
    using RetT = adt::Result<KernelArgId>;
    return val.Match(
        [](const DimExprKernelArgId<BirNode>& impl) -> RetT { return impl; },
        [](const InTensorDataPtrKernelArgId<BirNode>& impl) -> RetT {
          return impl;
        },
        [](const OutTensorDataPtrKernelArgId<BirNode>& impl) -> RetT {
          return impl;
        },
        [](const auto& impl) -> RetT {
          return adt::errors::TypeError{"KernelArgId::CastFrom() failed."};
        });
  }

  std::size_t GetHashValue() const {
    std::size_t hash_value = Match(
        [&](const auto& impl) -> std::size_t { return impl->GetHashValue(); });
    return adt::hash_combine(this->index(), hash_value);
  }
};

}  // namespace ap::code_gen

namespace std {

template <typename BirNode>
struct hash<ap::code_gen::KernelArgId<BirNode>> {
  std::size_t operator()(
      const ap::code_gen::KernelArgId<BirNode>& arg_id) const {
    return arg_id.GetHashValue();
  }
};

}  // namespace std
