#pragma once

#include "ck_patch/batched_matrix_coord.h"
#include "ck_patch/utility/data_type.hpp"
#include <type_traits>

namespace ck {
namespace tensor_operation {
namespace element_wise {

template <class VariadicOp, class = void>
struct GenericVariadicTraits {
  static constexpr bool IsArgumentsNeeded = false;
  struct Arguments {};
};

template <class VariadicOp>
struct GenericVariadicTraits<VariadicOp,
                             decltype(typename VariadicOp::Arguments(),
                                      void())> {
  static constexpr bool IsArgumentsNeeded = true;
  using Arguments = typename VariadicOp::Arguments;
};

template <
    class VariadicOp, 
    int ElementsPerAccess, 
    class = void>
struct VectorizedComputeTraits {
    static constexpr bool IsVectorized = false;
};

template <
    class VariadicOp, 
    int ElementsPerAccess>
struct VectorizedComputeTraits<VariadicOp, 
                               ElementsPerAccess,
                               decltype(std::declval<VariadicOp>().template Compute<ElementsPerAccess>(
                                   std::declval<typename VariadicOp::template OutVectorType<ElementsPerAccess>&>(),
                                   std::declval<typename VariadicOp::Arguments>(),
                                   std::declval<BatchedMatrixCoord>()), void())> {
    static constexpr bool IsVectorized = true;
};


template<
    template <typename T>
    class VariadicOp,
    typename ElementAccumulator_,
    int ElementsPerAccess
    >
class VariadicElementwiseOp
{   
public:
    static constexpr int kElementsPerAccess = ElementsPerAccess;

    using ElementAccumulator = ElementAccumulator_;
    using InnerVectorType = ck::vector_type<ElementAccumulator, kElementsPerAccess>;
    using VariadicArguments =
            typename GenericVariadicTraits<VariadicOp<ElementAccumulator>>::Arguments;
    
    static constexpr bool isVectorized = VectorizedComputeTraits<VariadicOp<ElementAccumulator>, kElementsPerAccess>::IsVectorized;

private:
    VariadicArguments arguments;

public:
    __host__ __device__ VariadicElementwiseOp(VariadicArguments _arguments) {
        arguments = _arguments;
    }

    __host__ __device__ void operator()(InnerVectorType& y, 
                                        const index_t batch,
                                        const index_t row_offset, 
                                        const index_t column_offset,
                                        bool is_valid) const
    {
        VariadicOp<ElementAccumulator> variadic_op;
        if constexpr(isVectorized) {
            y = variadic_op.template Compute<kElementsPerAccess>(y, arguments, 
                                                                 BatchedMatrixCoord(batch, row_offset, column_offset, is_valid));
        } else {
            ck::static_for<0, kElementsPerAccess, 1>{}([&](auto i) {
                y.template AsType<ElementAccumulator>()(i) = variadic_op(y.template AsType<ElementAccumulator>()(i), arguments, 
                                                                BatchedMatrixCoord(batch, row_offset, column_offset + i, is_valid));
            });
        }
    }
};


} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
