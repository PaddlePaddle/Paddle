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