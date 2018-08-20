// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <type_traits>
#include "unsupported/Eigen/CXX11/Tensor"

namespace Eigen {
namespace internal {
template <typename T>
struct index_op {
  using ResultType = T;
  static_assert(std::is_same<ResultType, uint32_t>::value ||
                    std::is_same<ResultType, int32_t>::value ||
                    std::is_same<ResultType, uint64_t>::value ||
                    std::is_same<ResultType, int64_t>::value,
                "Eigen::internal::index_op<T> error: T must be one of "
                "uint32_t, int32_t, uint64_t, int64_t");
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ResultType operator()() {
    return static_cast<ResultType>(0);
  }
};

template <typename T>
struct functor_traits<index_op<T>> {
  enum {
    Cost = NumTraits<T>::AddCost,
    PacketAccess = packet_traits<T>::HasAdd
  };
};
}  // namespace internal

#define DEFINE_TENSOR_INDEX_EVALUATOR(IndexType)                               \
  template <typename ArgType, typename Device>                                 \
  struct TensorEvaluator<                                                      \
      const TensorCwiseNullaryOp<internal::index_op<IndexType>, ArgType>,      \
      Device> {                                                                \
    typedef internal::index_op<IndexType> NullaryOp;                           \
    typedef TensorCwiseNullaryOp<NullaryOp, ArgType> XprType;                  \
                                                                               \
    enum {                                                                     \
      IsAligned = true,                                                        \
      PacketAccess = internal::functor_traits<NullaryOp>::PacketAccess,        \
      BlockAccess = false,                                                     \
      Layout = TensorEvaluator<ArgType, Device>::Layout,                       \
      CoordAccess = false,                                                     \
      RawAccess = false                                                        \
    };                                                                         \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE                                      \
    TensorEvaluator(const XprType &op, const Device &device)                   \
        : m_argImpl(op.nestedExpression(), device) {}                          \
    typedef typename XprType::Index Index;                                     \
    typedef typename XprType::Scalar Scalar;                                   \
    typedef typename internal::traits<XprType>::Scalar CoeffReturnType;        \
    typedef                                                                    \
        typename PacketType<CoeffReturnType, Device>::type PacketReturnType;   \
    static const int PacketSize =                                              \
        internal::unpacket_traits<PacketReturnType>::size;                     \
    typedef typename TensorEvaluator<ArgType, Device>::Dimensions Dimensions;  \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions &dimensions()       \
        const {                                                                \
      return m_argImpl.dimensions();                                           \
    }                                                                          \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(           \
        CoeffReturnType *) {                                                   \
      return true;                                                             \
    }                                                                          \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {}                    \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType                      \
    coeff(Index index) const {                                                 \
      return static_cast<CoeffReturnType>(index);                              \
    }                                                                          \
                                                                               \
    template <int LoadMode>                                                    \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType                     \
    packet(Index index) const {                                                \
      return internal::plset<PacketReturnType>(index);                         \
    }                                                                          \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost                         \
    costPerCoeff(bool vectorized) const {                                      \
      return TensorOpCost(sizeof(CoeffReturnType), 0, 0, vectorized,           \
                          internal::unpacket_traits<PacketReturnType>::size);  \
    }                                                                          \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr                            \
        typename Eigen::internal::traits<XprType>::PointerType                 \
        data() const {                                                         \
      return nullptr;                                                          \
    }                                                                          \
                                                                               \
    EIGEN_STRONG_INLINE const TensorEvaluator<ArgType, Device> &impl() const { \
      return m_argImpl;                                                        \
    }                                                                          \
                                                                               \
    EIGEN_STRONG_INLINE NullaryOp functor() const { return NullaryOp(); }      \
                                                                               \
   private:                                                                    \
    TensorEvaluator<ArgType, Device> m_argImpl;                                \
  }

DEFINE_TENSOR_INDEX_EVALUATOR(uint32_t);
DEFINE_TENSOR_INDEX_EVALUATOR(int32_t);
DEFINE_TENSOR_INDEX_EVALUATOR(uint64_t);
DEFINE_TENSOR_INDEX_EVALUATOR(int64_t);

#undef DEFINE_TENSOR_INDEX_EVALUATOR
}  // namespace Eigen
