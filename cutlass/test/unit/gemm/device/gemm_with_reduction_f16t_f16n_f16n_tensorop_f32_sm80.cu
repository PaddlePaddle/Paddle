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
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"

#include "cutlass/gemm/kernel/default_gemm_with_reduction.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/thread/linear_combination_drelu.h"
#include "cutlass/epilogue/thread/linear_combination_dgelu.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed_gemm_with_reduction.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

struct dReluLambda {
  float operator()(float d_y, float t) {
    if (t <= 0) {
      d_y = 0;
    }
    return d_y;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmWithReduction_dReLU_bGrad_f16t_f16n_f16n_tensor_op_f32, 128x128x32_64x64x32) {
  
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithReduction<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOutputOp,
      cutlass::plus<float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      5,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ReferenceOp = test::gemm::device::GemmWithReductionReference<
    Gemm, 
    dReluLambda
  >;

  test::gemm::device::TestGemmWithReduction<Gemm, ReferenceOp>(
    {136, 6920, 512},
    cutlass::gemm::GemmUniversalMode::kGemm
  );
}

#endif  // if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
