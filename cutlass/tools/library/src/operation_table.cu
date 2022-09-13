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
/*
  \file
  \brief Defines a data structure in which a set of functionally equivalent library::Operation
        instances may be queried.
*/

#include "cutlass/library/operation_table.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

void OperationTable::append(Manifest const &manifest) {

  // Insert operations into appropriate data structure
  for (auto const & operation : manifest) {

    OperationDescription const &desc = operation->description();

    // insert all gemm operation into operation table
    if (desc.kind == OperationKind::kGemm) {
      GemmDescription const &gemm_desc = static_cast<GemmDescription const &>(desc);
    

      GemmFunctionalKey functional_key(
        gemm_desc.provider,
        gemm_desc.gemm_kind,
        gemm_desc.tile_description.math_instruction.element_accumulator,
        gemm_desc.element_epilogue,
        gemm_desc.A.element,
        gemm_desc.A.layout,
        gemm_desc.transform_A,
        gemm_desc.B.element,
        gemm_desc.B.layout,
        gemm_desc.transform_B,
        gemm_desc.C.element
      );

      Operation const *op = operation.get();

      int cc = gemm_desc.tile_description.minimum_compute_capability;
        
      int alignment = std::max(std::max(
        gemm_desc.A.alignment, gemm_desc.B.alignment), gemm_desc.C.alignment);

      GemmPreferenceKey preference_key(cc, alignment);

      gemm_operations[functional_key][preference_key].push_back(op);
    }

    // insert all conv2d or conv3d operation into operation table
    if (desc.kind == OperationKind::kConv2d || desc.kind == OperationKind::kConv3d) {
      auto &conv_desc = static_cast<library::ConvDescription const &>(desc);

      ConvFunctionalKey functional_key(
        conv_desc.provider,
        conv_desc.conv_kind,
        conv_desc.A.element,
        conv_desc.A.layout,
        conv_desc.B.element,
        conv_desc.B.layout,
        conv_desc.C.element,
        conv_desc.C.layout,
        conv_desc.tile_description.math_instruction.element_accumulator, 
        conv_desc.element_epilogue
      );

      Operation const *op = operation.get();

      int cc = conv_desc.tile_description.minimum_compute_capability;

      ConvPreferenceKey preference_key(cc, conv_desc.iterator_algorithm);

      // insert conv operation to conv2d_operations or conv3d_operations map
      (desc.kind == OperationKind::kConv2d) ?
        conv2d_operations[functional_key][preference_key].push_back(op) : 
        conv3d_operations[functional_key][preference_key].push_back(op);
    }

    // insert all reduction operation into operation table
    if (desc.kind == OperationKind::kReduction) {
      auto &reduce_desc = static_cast<library::ReductionDescription const &>(desc);

      ReductionFunctionalKey functional_key(
        reduce_desc.provider,
        reduce_desc.element_workspace,
        reduce_desc.tile_description.math_instruction.element_accumulator,
        reduce_desc.element_output,
        reduce_desc.element_epilogue,
        library::MathOperationID::kAdd,
        library::EpilogueKind::kLinearCombination
      );

      Operation const *op = operation.get();

      reduction_operations[functional_key] = op;

    }

  }

}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

