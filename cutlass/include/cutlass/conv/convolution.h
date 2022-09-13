/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief 

This file contains definitions and utility functions for describing convolution problem sizes in terms of 
activation (NHWC), filter (KRSC), output (NPQK), pading (pad_h, pad_w), stride (stride_h, stride_w),
dilation (dilation_h, dilation_w).  Furthermore, it defines helper functions to map cutlass' implicit gemm 
tensor extents, sizes, data types to that of convolutions extents, sizes, and data types. 

                        * Mapping convolutions to Gemm computation *

Cutlass employs ImplicitGemm algorithm to implement convolutions. ImplicitGemm algorithm runs gemm operation 
on convolution tensors Activation, Filter, and Output . The underlying gemm operation follows the standard 
gemm definition:

                                     C = A * B + C

                               A and B are input matrices
                            C is source and output matrix


For the three convolutional operators (Fprop, Dgrad, Wgrad), ImplicitGemm matrices A, B, and C are mapped on 
to convolution tensors Activation, Filter and Output as per the below table:

        ___________________________________________________________________________
         ConvolutionalOperator |        A        |      B         |       C                           
        ___________________________________________________________________________
        |                      |                 |                |               |
        |       Fprop          |    Activation   |    Filter      |     Output    |  
        |       Dgrad          |     Output      |    Filter      |   Activation  |  
        |       Wgrad          |     Output      |  Activation    |     Filter    | 
        ___________________________________________________________________________

In convolution codebase, DO NOT mix using (A, B, C) with (Acvitation, Filter, Output).

For example, a convolution class/function with A, B, Output is confusing and error-prone. Instead use below 
mapping functions and adhere to using either A, B, C or Acvitation, Filter, Output. 

Map elements' data types (ImplicitGemm -> Conv): GemmToConvElementMap
Map elements' data types (Conv -> ImplicitGemm): ConvToGemmElementMap
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

namespace cutlass {
namespace conv {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Convolutional operator
enum class Operator { 
  kFprop, 
  kDgrad, 
  kWgrad 
};

/// Distinguishes convolution  from cross correlation
enum class Mode { 
  kCrossCorrelation, 
  kConvolution 
};

/// Selects among several implementation variants trading off performance with simplicity
enum class IteratorAlgorithm { 
  kAnalytic,      ///< functionally correct in all cases but lower performance
  kOptimized,     ///< optimized for R <= 32, S <= 32 and unity-stride dgrad
  kFixedChannels, ///< Analytic algorithm optimized for fixed channel count (C == AccessSize)
  kFewChannels    ///< Analytic algorithm optimized for few channels (C divisible by AccessSize)
};

/// Distinguishes among partial specializations that accelerate certain problems where convolution
/// stride is unit.
enum class StrideSupport {
  kStrided,       ///< arbitrary convolution stride
  kUnity          ///< unit convolution stride
};

/// Identifies split-K mode
enum class SplitKMode { 
  kNone, 
  kSerial, 
  kParallel
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace conv
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
