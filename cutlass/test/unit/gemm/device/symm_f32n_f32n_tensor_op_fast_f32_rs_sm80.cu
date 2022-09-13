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
    \brief Tests for device-wide SYMM interface

  
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/blas3.h"
#include "cutlass/gemm/device/symm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/symm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_symm_universal.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////Test name//////////////////////////////////////////////////
//                             
// SM80_Device_Symm_{ElementA/B}{LayoutA/B}_{ElementC}{LayoutC}_{SideMode}_{FillMode}\
//     _tensor_op_{ElementAccumulator}_align{AlignmentA}_align{AlignmentB}
//
///////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Symm_f32n_f32n_rs_u_tensor_op_fast_f32_align1_align1, 64x128x32_32x64x32) {

using Symm = cutlass::gemm::device::Symm<
    float,
    cutlass::layout::ColumnMajor,
    cutlass::SideMode::kRight, 
    cutlass::FillMode::kUpper,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddFastF32
>;

  EXPECT_TRUE(test::gemm::device::TestAllSymmUniversal<Symm>());
}
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Symm_f32n_f32n_rs_u_tensor_op_fast_f32_align1_align1, 128x64x32_32x64x32) {

using Symm = cutlass::gemm::device::Symm<
    float,
    cutlass::layout::ColumnMajor,
    cutlass::SideMode::kRight, 
    cutlass::FillMode::kUpper,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddFastF32
>;

  EXPECT_TRUE(test::gemm::device::TestAllSymmUniversal<Symm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Symm_f32n_f32n_rs_l_tensor_op_fast_f32_align1_align1, 64x128x32_32x64x32) {

using Symm = cutlass::gemm::device::Symm<
    float,
    cutlass::layout::ColumnMajor,
    cutlass::SideMode::kRight, 
    cutlass::FillMode::kLower,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAddFastF32
>;

  EXPECT_TRUE(test::gemm::device::TestAllSymmUniversal<Symm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////


TEST(SM80_Device_Symm_f32n_f32n_rs_u_tensor_op_fast_f32_align1_align4, 64x128x32_32x64x32) {

using Symm = cutlass::gemm::device::Symm<
    float,
    cutlass::layout::ColumnMajor,
    cutlass::SideMode::kRight, 
    cutlass::FillMode::kUpper,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
>;

  EXPECT_TRUE(test::gemm::device::TestAllSymmUniversal<Symm>());
}
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Symm_f32n_f32n_rs_u_tensor_op_fast_f32_align1_align4, 128x64x32_32x64x32) {

using Symm = cutlass::gemm::device::Symm<
    float,
    cutlass::layout::ColumnMajor,
    cutlass::SideMode::kRight, 
    cutlass::FillMode::kUpper,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
>;

  EXPECT_TRUE(test::gemm::device::TestAllSymmUniversal<Symm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Symm_f32n_f32n_rs_l_tensor_op_fast_f32_align1_align4, 64x128x32_32x64x32) {

using Symm = cutlass::gemm::device::Symm<
    float,
    cutlass::layout::ColumnMajor,
    cutlass::SideMode::kRight, 
    cutlass::FillMode::kLower,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    1,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
>;

  EXPECT_TRUE(test::gemm::device::TestAllSymmUniversal<Symm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
#endif // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
