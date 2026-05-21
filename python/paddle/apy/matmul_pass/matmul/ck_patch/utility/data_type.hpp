// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "ck/utility/data_type.hpp"
#include <hip/hip_bfloat16.h> 

namespace ck {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// Convert HIP data type to ck data type
template <typename T>
struct CKDataType {
  using Type = T;
};

template <>
struct CKDataType<half> {
  using Type = ck::half_t;
};

template <>
struct CKDataType<hip_bfloat16> {
  using Type = ck::bhalf_t;
};

// T native type 
template <typename T, int VecSize>
struct VectorType {
    using CKType = typename ck::CKDataType<T>::Type;
    using InnerVectorType = ck::vector_type<CKType, VecSize>;
    
    InnerVectorType& inner_vector;

    __host__ __device__ constexpr VectorType() : inner_vector() {}

    __host__ __device__ constexpr VectorType(InnerVectorType& v) : inner_vector(v) {}

    template <int I>
    __host__ __device__ constexpr const auto& operator()(Number<I> i) const {
        return inner_vector.template AsType<CKType>()(i);
    }

    template <int I>
    __host__ __device__ constexpr auto& operator()(Number<I> i) {
        return inner_vector.template AsType<CKType>()(i);
    }
};

} // namespace ck