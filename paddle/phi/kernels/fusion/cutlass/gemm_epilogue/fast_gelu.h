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

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator followed by the GELU activation to an
/// array of elements.
///
/// D = gelu(alpha * accumulator + beta * source + uniform)
///
template <
    typename ElementOutput_,  ///< Data type used to load and store tensors
    int Count,                ///< Number of elements computed per operation
                ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                ///< but we use 64 or 32 sometimes when there are not enough
                ///< data to store
    typename ElementAccumulator_ = ElementOutput_,  ///< Accumulator data type
    typename ElementCompute_ =
        ElementOutput_,  ///< Data type used to compute linear combination
    ScaleType::Kind Scale =
        ScaleType::Default,  ///< Control Alpha and Beta scaling
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
using LinearCombinationFastGELU = LinearCombinationGeneric<GELU_taylor,
                                                           ElementOutput_,
                                                           Count,
                                                           ElementAccumulator_,
                                                           ElementCompute_,
                                                           Scale,
                                                           Round,
                                                           true>;

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass
