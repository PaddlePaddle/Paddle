#pragma once

#include "cpu_patch/batched_matrix_coord.h"
#include "cpu_patch/data_type.h"
#include "cpu_patch/unroll.h"
#include <stdexcept>
#include <type_traits>

namespace gops {
namespace epilogue {

template <class VariadicOp, class = void>
struct VariadicArgumentTraits {
  static constexpr bool IsArgumentsNeeded = false;
  struct Arguments {};
};

template <class VariadicOp>
struct VariadicArgumentTraits<VariadicOp,
                              decltype(typename VariadicOp::Arguments(), void())> {
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
    int ElementsPerAccess,
    typename ElementAccumulator_>
class VariadicEpilogue {   
public:
    using ElementAccumulator = ElementAccumulator_;
    static constexpr int kElementsPerAccess = ElementsPerAccess;
    static constexpr bool isVectorized = VectorizedComputeTraits<VariadicOp<ElementAccumulator>, kElementsPerAccess>::IsVectorized;
    
    using VariadicArguments =
            typename VariadicArgumentTraits<VariadicOp<ElementAccumulator>>::Arguments;
    using InnerVectorType = VectorType<ElementAccumulator, kElementsPerAccess>;
    
private:
    VariadicArguments arguments;

public:
    VariadicEpilogue(VariadicArguments _arguments) {
        arguments = _arguments;
    }

    void operator()(InnerVectorType& y,
                    int batch,
                    int row_offset, 
                    int column_offset,
                    int valid) const
    {
        VariadicOp<ElementAccumulator> variadic_op;
        if constexpr(isVectorized) {
            // kernel is col-major but interface is row-major
            y = variadic_op.template Compute<kElementsPerAccess>(y, arguments, 
                                                                 BatchedMatrixCoord(batch, column_offset, row_offset, valid)); 
        } else {
            unroll<kElementsPerAccess>{}([&](auto i) {
                y[i.value] = variadic_op(y[i.value], arguments,
                                BatchedMatrixCoord(batch, column_offset, row_offset + i, valid));
            });
        }
    }
};


struct Passthrough {};

}; // namespace epilogue
}; // namespace gops


