#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {

template<
  typename TensorLayout_,                             ///! The original output tensor layout
  typename OutputIteratorLayout_,                     ///! Layout used by epilogue output iterator
  typename TensorRef_,                                ///! Input tensor to epilogue output iterator
  conv::Operator ConvOperator,                        ///! Convolutional operator (Fprop, Dgrad, Wgrad)
  typename ConvProblemSize_                          ///! Convolutional operator on 2D or 3D problem
>
struct ConvOutputIteratorParameter {

  using TensorLayout = TensorLayout_;
  using OutputIteratorLayout = OutputIteratorLayout_;
  using OutputTensorCoord = typename OutputIteratorLayout::TensorCoord;
  using TensorRef = TensorRef_;
  static conv::Operator const kConvolutionalOperator = ConvOperator;
  using ConvProblemSize = ConvProblemSize_;

  /// Wgrad stride idx for implicit gemm algorithm 
  // Conv2d row-major matrix (KxRSC) 
  // Conv3d row-major matrix (KxTRSC)
  static int const kWgradStrideIdx = 
    platform::is_same<TensorLayout, layout::TensorNHWC>::value ? 2 : 3;

  /// This chooses the appropriate stride element of the C tensor.
  static int const kTensorStrideIdx = 
    (kConvolutionalOperator == conv::Operator::kWgrad ? kWgradStrideIdx : 0);


  CUTLASS_HOST_DEVICE
  static OutputIteratorLayout layout(const TensorRef & ref) {
    return ref.stride(kTensorStrideIdx);
  }

  CUTLASS_HOST_DEVICE
  static OutputTensorCoord extent(ConvProblemSize problem_size) {
    return conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn();
  }

};



template <
  int InterleavedK,
  typename TensorRef_,
  conv::Operator ConvOperator,
  typename ConvProblemSize_
>
struct ConvOutputIteratorParameter<
  layout::TensorNCxHWx<InterleavedK>, 
  layout::TensorNCxHWx<InterleavedK>,
  TensorRef_,
  ConvOperator,
  ConvProblemSize_>
{ 

  using TensorLayout = typename layout::TensorNCxHWx<InterleavedK>;
  using OutputIteratorLayout = typename layout::TensorNCxHWx<InterleavedK>;
  using OutputTensorCoord = typename OutputIteratorLayout::TensorCoord;
  using TensorRef = TensorRef_;
  static conv::Operator const kConvolutionalOperator = ConvOperator;
  using ConvProblemSize = ConvProblemSize_;

  CUTLASS_HOST_DEVICE
  static OutputIteratorLayout layout(const TensorRef & ref) {
    return ref.stride();
  }

  CUTLASS_HOST_DEVICE
  static OutputTensorCoord extent(ConvProblemSize problem_size) {
    return problem_size.output_extent();
  }

};

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass
