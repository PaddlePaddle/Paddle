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
    \brief Tests for device-wide SYRK interface
  
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/blas3.h"
#include "cutlass/gemm/device/rank_k.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/rank_k_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_rank_k_universal.h"

#if (CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_l_tensor_op_f32, 128x256x32_64x64x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kLower,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_l_tensor_op_f32, 256x128x32_64x64x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kLower,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_l_tensor_op_f32, 64x256x32_64x64x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kLower,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_l_tensor_op_f32, 256x64x32_64x64x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kLower,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_l_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kLower,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

TEST(SM80_Device_Syrk_tf32t_f32t_l_tensor_op_f32, 64x128x32_32x64x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kLower,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

TEST(SM80_Device_Syrk_tf32t_f32t_l_tensor_op_f32, 128x64x32_64x32x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kLower,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_l_tensor_op_f32, 128x128x16_64x64x16) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kLower,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_l_tensor_op_f32, 64x128x16_32x64x16) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kLower,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    6
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());

}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_u_tensor_op_f32, 128x256x32_64x64x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kUpper,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_u_tensor_op_f32, 256x128x32_64x64x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kUpper,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_u_tensor_op_f32, 64x256x32_64x64x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kUpper,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_u_tensor_op_f32, 256x64x32_64x64x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kUpper,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syrk_tf32t_f32t_u_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementA = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using RankK = cutlass::gemm::device::RankK<
    ElementA,
    LayoutA,
    ElementC,
    LayoutC,
    cutlass::FillMode::kUpper,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRankKUniversal<RankK>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif // if (CUTLASS_ARCH_MMA_SM80_SUPPORTED)
