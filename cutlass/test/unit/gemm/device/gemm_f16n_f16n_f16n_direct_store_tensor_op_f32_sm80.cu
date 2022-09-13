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

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_universal.h"

////////////////////////////////////////////////////////////////////////////////

#include "cutlass/epilogue/threadblock/epilogue_direct_store.h"
#include "cutlass/epilogue/threadblock/default_epilogue_direct_store.h"

////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_GemmUniversal_DirectStore_f16n_f16t_f32n_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  // Define the GEMM kernel
  using GemmBase = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t, 
      cutlass::layout::ColumnMajor, 
      cutlass::half_t,
      cutlass::layout::RowMajor, 
      ElementOutput, cutlass::layout::ColumnMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>, 
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 
          4,                            // This is the vector size of the epilogue. 
          ElementAccumulator, 
          ElementAccumulator>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
    3,
    8,
    8
  >;

  // Define the direct store epilogue
  using EpilogueDirectStore = typename cutlass::epilogue::threadblock::DefaultEpilogueDirectStore<
    typename GemmBase::GemmKernel::Epilogue
  >::Epilogue;

  // Define a new kernel
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    typename GemmBase::GemmKernel::Mma,
    EpilogueDirectStore,
    typename GemmBase::GemmKernel::ThreadblockSwizzle
  >;

  // Define the adaptor
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  EXPECT_TRUE(test::gemm::device::TestAllGemmUniversal<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

#endif  // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

