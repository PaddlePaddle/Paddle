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
  \brief Defines structures and helpers to launch CUDA kernels within CUTLASS.
*/

#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure containing the basic launch configuration of a CUDA kernel.
struct KernelLaunchConfiguration {

  /// CUDA grid dimensions
  dim3 grid;

  /// CUDA threablock dimensions
  dim3 block;

  /// Bytes of dynamically allocated SMEM in addition to static SMEM
  size_t dynamic_smem;

  //
  // Methods
  //

  /// Constructs a KernellaunchConfiguration object
  CUTLASS_HOST_DEVICE
  KernelLaunchConfiguration(
    dim3 _grid = dim3(1,1,1),
    dim3 _block = dim3(1,1,1),
    size_t _dynamic_smem = 0
  ):
    grid(_grid),
    block(_block),
    dynamic_smem(_dynamic_smem) { }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
