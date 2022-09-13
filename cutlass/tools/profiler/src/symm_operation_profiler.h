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
/* \file
   \brief Defines a math function

  
*/

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <unordered_map>

// CUTLASS Library includes
#include "cutlass/blas3.h"
#include "cutlass/library/library.h"
#include "cutlass/library/util.h"
#include "cutlass/library/manifest.h"

// Profiler includes
#include "options.h"
#include "device_context.h"
#include "operation_profiler.h"
#include "performance_result.h"
#include "problem_space.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////


/// Abstract base class for each math function
class SymmOperationProfiler : public OperationProfiler {
public:

  /// Problem structure obtained from problem space
  struct SymmProblem {
    int64_t m;
    int64_t n;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    SideMode side_mode;
    FillMode fill_mode;
    BlasMode blas_mode;
    std::vector<uint8_t> alpha;
    std::vector<uint8_t> beta;
    int64_t split_k_slices;
    int64_t batch_count;

    //
    // Methods
    //

    SymmProblem(): 
      m(16), n(16), lda(0), ldb(0), ldc(0), 
      side_mode(SideMode::kInvalid), fill_mode(FillMode::kInvalid), blas_mode(BlasMode::kInvalid), 
      split_k_slices(1), batch_count(1) { }

    /// Parses the problem
    Status parse(
      library::SymmDescription const &operation_desc,
      ProblemSpace const &problem_space,
      ProblemSpace::Problem const &problem);

    /// Total number of bytes loaded
    int64_t bytes(library::SymmDescription const &operation_desc) const;

    /// Total number of flops computed
    int64_t flops(library::SymmDescription const &operation_desc) const;
    
    /// Initializes a performance result
    void initialize_result(
      PerformanceResult &result,
      library::SymmDescription const &operation_desc,
      ProblemSpace const &problem_space);
  };

  /// Workspace used 
  struct SymmWorkspace {

    DeviceAllocation *A;
    DeviceAllocation *B;
    DeviceAllocation *C;
    DeviceAllocation *Computed;
    DeviceAllocation *Reference;

    library::SymmConfiguration configuration;
    library::SymmArguments arguments;

    /// Buffer used for the operation's host workspace
    std::vector<uint8_t> host_workspace;

    /// Buffer used for the operations' device workspace
    DeviceAllocation device_workspace;

    //
    // Methods
    //

    SymmWorkspace(): 
      A(nullptr), B(nullptr), C(nullptr), Computed(nullptr), Reference(nullptr) { }
  };

protected:

  //
  // Data members
  //

  /// GEMM problem obtained from problem space
  SymmProblem problem_;

  /// Device memory allocations 
  SymmWorkspace symm_workspace_;


public:
  //
  // Methods
  //

  /// Ctor
  SymmOperationProfiler(Options const &options);

  /// Destructor
  virtual ~SymmOperationProfiler();

  /// Prints usage statement for the math function
  virtual void print_usage(std::ostream &out) const;

  /// Prints examples
  virtual void print_examples(std::ostream &out) const;

  /// Extracts the problem dimensions
  virtual Status initialize_configuration(
    Options const &options, 
    PerformanceReport &report, 
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

  /// Initializes workspace
  virtual Status initialize_workspace(
    Options const &options, 
    PerformanceReport &report, 
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

  /// Verifies CUTLASS against references
  virtual bool verify_cutlass(
    Options const &options,  
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

  /// Measures performance results
  virtual bool profile(
    Options const &options, 
    PerformanceReport &report, 
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

protected:

  /// Initializes the performance result
  void initialize_result_(
    PerformanceResult &result,
    Options const &options,  
    library::SymmDescription const &operation_desc,
    ProblemSpace const &problem_space);

  /// Verifies CUTLASS against references
  bool verify_with_cublas_(
    Options const &options,  
    PerformanceReport &report,
    DeviceContext &device_context,
    library::Operation const *operation,
    ProblemSpace const &problem_space,
    ProblemSpace::Problem const &problem);

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
