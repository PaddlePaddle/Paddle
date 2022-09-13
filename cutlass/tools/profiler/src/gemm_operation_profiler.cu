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
   \brief Execution environment
*/

#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <ios>

#include "cutlass/core_io.h"

#include "cublas_helpers.h"
#include "gemm_operation_profiler.h"
#include "gpu_timer.h"

#include "cutlass/library/singleton.h"
#include "cutlass/library/library.h"
#include "cutlass/library/handle.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Ctor
GemmOperationProfiler::GemmOperationProfiler(Options const &options): 
  OperationProfiler(
    options,
    library::OperationKind::kGemm,
    {
      {ArgumentTypeID::kEnumerated, {"gemm_kind"}, "Variant of GEMM (gemm, batched, array, universal, planar_complex, planar_complex_array)"},
      {ArgumentTypeID::kEnumerated, {"split_k_mode"}, "Variant of split K mode(serial, parallel)"},
      {ArgumentTypeID::kInteger, {"m", "problem-size::m"}, "M dimension of the GEMM problem space"},
      {ArgumentTypeID::kInteger, {"n", "problem-size::n"}, "N dimension of the GEMM problem space"},
      {ArgumentTypeID::kInteger, {"k", "problem-size::k"}, "K dimension of the GEMM problem space"},
      {ArgumentTypeID::kTensor, {"A"}, "Tensor storing the A operand"},
      {ArgumentTypeID::kTensor, {"B"}, "Tensor storing the B operand"},
      {ArgumentTypeID::kTensor, {"C"}, "Tensor storing the C operand"},
      {ArgumentTypeID::kScalar, {"alpha", "epilogue::alpha"}, "Epilogue scalar alpha"},
      {ArgumentTypeID::kScalar, {"beta", "epilogue::beta"}, "Epilogue scalar beta"},
      {ArgumentTypeID::kInteger, {"split_k_slices", "split-k-slices"}, "Number of partitions of K dimension"},
      {ArgumentTypeID::kInteger, {"batch_count", "batch-count"}, "Number of GEMMs computed in one batch"},
    },
    { library::Provider::kCUBLAS}
  ) {

  description_ = "      General matrix-matrix product. D = alpha * A*B + beta * C";
}

/// Destructor
GemmOperationProfiler::~GemmOperationProfiler() {

}

/// Prints usage statement for the math function
void GemmOperationProfiler::print_usage(std::ostream &out) const {
  out << "GEMM" << "\n\n";

  OperationProfiler::print_usage(out);
}

/// Prints examples
void GemmOperationProfiler::print_examples(std::ostream &out) const {

  out << "\nExamples:\n\n"
    << "Profile a particular problem size:\n"
    << "  $ cutlass_profiler --operation=Gemm --m=1024 --n=1024 --k=128\n\n"

    << "Schmoo over problem size and beta:\n"
    << "  $ cutlass_profiler --operation=Gemm --m=1024:4096:256 --n=1024:4096:256 --k=128:8192:128 --beta=0,1,2.5\n\n"

    << "Schmoo over accumulator types:\n"
    << "  $ cutlass_profiler --operation=Gemm --accumulator-type=f16,f32\n\n"

    << "Run when A is f16 with column-major and B is any datatype with row-major (For column major, use column, col, or n. For row major use, row or t):\n"
    << "  $ cutlass_profiler --operation=Gemm --A=f16:column --B=*:row\n\n"

    << "Profile a particular problem size with split K and paralell reduction:\n"
    << "  $ cutlass_profiler --operation=Gemm --split_k_mode=parallel --split_k_slices=2 --m=1024 --n=1024 --k=128\n\n"

    << "Using various input value distribution:\n"
    << "  $ cutlass_profiler --operation=Gemm --dist=uniform,min:0,max:3\n"
    << "  $ cutlass_profiler --operation=Gemm --dist=gaussian,mean:0,stddev:3\n"
    << "  $ cutlass_profiler --operation=Gemm --dist=sequential,start:0,delta:1\n\n"

    << "Run a kernel with cta tile size of 256x128x32 and save workspace if results are incorrect (note that --cta-tile::k=32 is default cta-tile size):\n"
    << " $ cutlass_profiler --operation=Gemm --cta_m=256 --cta_n=128  --cta_k=32 --save-workspace=incorrect\n\n"
    
    << "Test your changes to gemm kernels with a quick functional test and save results in functional-test.csv:\n"
    << " $ cutlass_profiler  --operation=Gemm \\ \n"
    << "   --m=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \\ \n"
    << "   --n=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \\ \n"
    << "   --k=8,16,32,64,128,256,288,384,504,512,520 \\ \n"
    << "   --beta=0,1,2 --profiling-iterations=1 \\ \n"
    << "   --providers=cutlass --output=functional-test.csv\n\n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
// used this for debugging
static std::string byte_string(std::vector<uint8_t> const &bytes) {
  std::stringstream ss;

  ss << "0x";

  for (size_t idx = bytes.size(); idx > 0; --idx) {
    ss << std::hex << std::setw(2) << std::setfill('0') << uint32_t(bytes.at(idx - 1));
  }

  return ss.str();
}
#endif

Status GemmOperationProfiler::GemmProblem::parse(
  library::GemmDescription const &operation_desc,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {
    
  this->mode = library::GemmUniversalMode::kGemm;
  
  if (!arg_as_int(this->m, "m", problem_space, problem)) {
    // default value
    this->m = 1024;
  }

  if (!arg_as_int(this->n, "n", problem_space, problem)) {
    // default value
    this->n = 1024;
  }
  
  if (!arg_as_int(this->k, "k", problem_space, problem)) {
    // default value
    this->k = 1024;
  }

  if (!arg_as_SplitKModeID(this->split_k_mode, "split_k_mode", problem_space, problem)) {
    // defualt value
    this->split_k_mode = library::SplitKMode::kSerial;
  }
  
  this->mode = library::GemmUniversalMode::kGemm;
  if(this->split_k_mode == library::SplitKMode::kParallel) {
    this->mode = library::GemmUniversalMode::kGemmSplitKParallel;
  }

  if (!arg_as_int(this->split_k_slices, "split_k_slices", problem_space, problem)) {
    // default value
    this->split_k_slices = 1;
  }
  
  if (!arg_as_int(this->batch_count, "batch_count", problem_space, problem)) {
    // default value
    this->batch_count = 1;
  } else if (this->batch_count > 1) {
    this->mode = library::GemmUniversalMode::kBatched;
  }

  if (this->split_k_slices > 1 && this->batch_count > 1) {
    // At least one of these must be one
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.A, "A", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.B, "B", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.C, "C", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!arg_as_scalar(
    this->alpha, 
    operation_desc.element_epilogue, 
    "alpha", 
    problem_space, 
    problem)) {

    if (!cast_from_double(this->alpha, operation_desc.element_epilogue, 1)) {
      return Status::kErrorInternal;
    }
  }
  
  if (!arg_as_scalar(
    this->beta, 
    operation_desc.element_epilogue, 
    "beta", 
    problem_space, 
    problem)) {
    
    if (!cast_from_double(this->beta, operation_desc.element_epilogue, 0)) {
      return Status::kErrorInternal;
    }
  }

  this->lda = DeviceAllocation::get_packed_layout(
    operation_desc.A.layout, {int(this->m), int(this->k)}).front();

  this->ldb = DeviceAllocation::get_packed_layout(
    operation_desc.B.layout, {int(this->k), int(this->n)}).front();

  this->ldc = DeviceAllocation::get_packed_layout(
    operation_desc.C.layout, {int(this->m), int(this->n)}).front();

  return Status::kSuccess;
}

/// Total number of bytes loaded
int64_t GemmOperationProfiler::GemmProblem::bytes(library::GemmDescription const &operation_desc) const {
  // Input bytes read and Output bytes written for the gemm problem
  int64_t bytes =
    int64_t(library::sizeof_bits(operation_desc.A.element) * m / 8) * k +
    int64_t(library::sizeof_bits(operation_desc.B.element) * n / 8) * k +
    int64_t(library::sizeof_bits(operation_desc.C.element) * m / 8) * n;

  // Set is_beta_zero true if beta is zero
  bool is_beta_zero = std::all_of(beta.begin(), beta.end(), [](uint8_t i) { return i==0; });

  // Output bytes read for the gemm problem for non-zero beta values
  if (!is_beta_zero) {
    bytes += int64_t(library::sizeof_bits(operation_desc.C.element) * m / 8) * n;
  }

  bytes *= batch_count;

  return bytes;
}

/// Total number of flops computed
int64_t GemmOperationProfiler::GemmProblem::flops(library::GemmDescription const &operation_desc) const {
  int64_t flops_ = (int64_t(m) * n * k + m * n) * 2 * batch_count;

  // complex-valued support
  switch (operation_desc.tile_description.math_instruction.math_operation) {
  case library::MathOperationID::kMultiplyAddComplex:
    flops_ *= 4;
    break;

  case library::MathOperationID::kMultiplyAddComplexFastF32:
    flops_ *= 4;
    break;

  case library::MathOperationID::kMultiplyAddGaussianComplex:
    flops_ *= 3;
    break;

  default: break;
  }

  return flops_;
}


/// Initializes a performance result
void GemmOperationProfiler::GemmProblem::initialize_result(
  PerformanceResult &result,
  library::GemmDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  result.arguments.resize(problem_space.rank());

  set_argument(result, "gemm_kind", problem_space, library::to_string(operation_desc.gemm_kind));

  set_argument(result, "split_k_mode", problem_space, library::to_string(split_k_mode));

  set_argument(result, "A", problem_space,
    std::string(library::to_string(operation_desc.A.element)) + ":" + library::to_string(operation_desc.A.layout));

  set_argument(result, "B", problem_space,
    std::string(library::to_string(operation_desc.B.element)) + ":" + library::to_string(operation_desc.B.layout));

  set_argument(result, "C", problem_space,
    std::string(library::to_string(operation_desc.C.element)) + ":" + library::to_string(operation_desc.C.layout));

  set_argument(result, "m", problem_space, m);
  set_argument(result, "n", problem_space, n);
  set_argument(result, "k", problem_space, k);

  set_argument(result, "split_k_slices", problem_space, split_k_slices);
  set_argument(result, "batch_count", problem_space, batch_count);

  set_argument(result, "alpha", problem_space,
    library::lexical_cast(alpha, operation_desc.element_epilogue));

  set_argument(result, "beta", problem_space,
    library::lexical_cast(beta, operation_desc.element_epilogue));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Extracts the problem dimensions
Status GemmOperationProfiler::initialize_configuration(
  Options const &options,  
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::GemmDescription const &operation_desc = 
    static_cast<library::GemmDescription const &>(operation->description());

  if (operation_desc.gemm_kind != library::GemmKind::kUniversal) {
    return Status::kErrorInvalidProblem;
  }

  Status status = problem_.parse(operation_desc, problem_space, problem);

  if (status != Status::kSuccess) {
    return status;
  }

  gemm_workspace_.configuration.mode = problem_.mode;
  gemm_workspace_.configuration.problem_size.m() = int(problem_.m);
  gemm_workspace_.configuration.problem_size.n() = int(problem_.n);
  gemm_workspace_.configuration.problem_size.k() = int(problem_.k);
  gemm_workspace_.configuration.lda = problem_.lda;
  gemm_workspace_.configuration.ldb = problem_.ldb;
  gemm_workspace_.configuration.ldc = problem_.ldc;
  gemm_workspace_.configuration.ldd = problem_.ldc;

  if (problem_.mode == library::GemmUniversalMode::kBatched) {
    gemm_workspace_.configuration.batch_count = problem_.batch_count;
  }
  else {
    gemm_workspace_.configuration.batch_count = problem_.split_k_slices;
  }

  gemm_workspace_.arguments.A = nullptr;
  gemm_workspace_.arguments.B = nullptr;
  gemm_workspace_.arguments.C = nullptr;
  gemm_workspace_.arguments.D = nullptr;
  gemm_workspace_.arguments.alpha = problem_.alpha.data();
  gemm_workspace_.arguments.beta = problem_.beta.data();
  gemm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  // initialize reduction operation for parallel splitKMode
  if (problem_.split_k_mode == library::SplitKMode::kParallel) {
    if (!initialize_reduction_configuration_(operation, problem)) {
      return Status::kErrorInternal;
    }
  }

  initialize_result_(this->model_result_, options, operation_desc, problem_space);
  
  return operation->can_implement(&gemm_workspace_.configuration, &gemm_workspace_.arguments);
}

/// Initializes the performance result
void GemmOperationProfiler::initialize_result_(
  PerformanceResult &result,
  Options const &options,  
  library::GemmDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  result.provider = library::Provider::kCUTLASS;
  result.disposition = Disposition::kNotRun;
  result.status = Status::kSuccess;
  result.operation_name = operation_desc.name;

  problem_.initialize_result(result, operation_desc, problem_space);

  OperationProfiler::initialize_result_(result, operation_desc, problem_space);

  result.bytes = problem_.bytes(operation_desc);
  result.flops = problem_.flops(operation_desc);
  result.runtime = 0;

}

/// Initialize redution problem dimentions and library::Operation
bool GemmOperationProfiler::initialize_reduction_configuration_(
  library::Operation const *operation,
  ProblemSpace::Problem const &problem) {
  library::GemmDescription const &gemm_desc =
    static_cast<library::GemmDescription const&>(operation->description());

  if (!cast_from_double(problem_.alpha_one, gemm_desc.element_epilogue, 1)) {
    return false;
  }

  if (!cast_from_double(problem_.beta_zero, gemm_desc.element_epilogue, 0)) {
    return false;
  }

  /// initialize library::ReductionConfiguration
  gemm_workspace_.reduction_configuration.problem_size      = gemm::GemmCoord(int(problem_.n), int(problem_.m), int(problem_.k)).mn();
  gemm_workspace_.reduction_configuration.partitions        = int(problem_.split_k_slices);
  gemm_workspace_.reduction_configuration.partition_stride  = gemm::GemmCoord(int(problem_.n), int(problem_.m), int(problem_.k)).mn().product();
  gemm_workspace_.reduction_configuration.ldw               = problem_.ldc;
  gemm_workspace_.reduction_configuration.lds               = problem_.ldc;
  gemm_workspace_.reduction_configuration.ldd               = problem_.ldc;

  // find reduction operation
  library::ReductionFunctionalKey reduction_key(
    library::Provider::kCUTLASS,
    gemm_desc.tile_description.math_instruction.element_accumulator,    // element workspace
    gemm_desc.tile_description.math_instruction.element_accumulator,    // element accumulator
    gemm_desc.C.element,                                                // element output
    gemm_desc.element_epilogue                                          // element coumpute
  );

  auto reduction_it = library::Singleton::get().operation_table.reduction_operations.find(reduction_key);
 
  if (reduction_it == library::Singleton::get().operation_table.reduction_operations.end()) {
    return false;
  }

  // initialize reduction operation required for parallel split-k operator
  reduction_op_ = reduction_it->second;

  // reduction operation found and initialized
  return true;
}

/// Initializes workspace
Status GemmOperationProfiler::initialize_workspace(
  Options const &options,  
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::Operation const* underlying_operation = operation;

  if (problem_.split_k_mode == library::SplitKMode::kParallel) {
    if (!(underlying_operation = library::find_gemm_operation_for_parallel_reduction(operation))) {
      return Status::kErrorNotSupported;
    }
  }

  library::GemmDescription const &operation_desc = 
    static_cast<library::GemmDescription const &>(operation->description());

  // Compute the number of copies of the problem to avoid L2 camping.
  if (!options.profiling.workspace_count) {
    int64_t bytes = problem_.bytes(operation_desc);
    if (bytes < 3 * int64_t(options.device.properties.l2CacheSize)) {
      gemm_workspace_.problem_count = 
        1 + int((3 * int64_t(options.device.properties.l2CacheSize)) / bytes);
    }
    else {
      gemm_workspace_.problem_count = 1;
    }
  }
  else {
    gemm_workspace_.problem_count = options.profiling.workspace_count;
  }

  if (options.execution_mode != ExecutionMode::kDryRun) {

    gemm_workspace_.A = device_context.allocate_tensor(
      options,
      "A",
      operation_desc.A.element,
      operation_desc.A.layout,
      {int(problem_.m), int(problem_.k)},
      {int(problem_.lda)},
      problem_.batch_count * gemm_workspace_.problem_count
    );

    gemm_workspace_.B = device_context.allocate_tensor(
      options,
      "B",
      operation_desc.B.element,
      operation_desc.B.layout,
      {int(problem_.k), int(problem_.n)},
      {int(problem_.ldb)},
      problem_.batch_count * gemm_workspace_.problem_count
    );

    gemm_workspace_.C = device_context.allocate_tensor(
      options,
      "C",
      operation_desc.C.element,
      operation_desc.C.layout,
      {int(problem_.m), int(problem_.n)},
      {int(problem_.ldc)},
      problem_.batch_count * gemm_workspace_.problem_count
    );

    gemm_workspace_.Computed = device_context.allocate_tensor(
      "D",
      operation_desc.C.element,
      operation_desc.C.layout,
      {int(problem_.m), int(problem_.n)},
      {int(problem_.ldc)},
      problem_.batch_count * gemm_workspace_.problem_count
    );

    gemm_workspace_.Reference = device_context.allocate_tensor(
      "Reference",
      operation_desc.C.element,
      operation_desc.C.layout,
      {int(problem_.m), int(problem_.n)},
      {int(problem_.ldc)},
      problem_.batch_count * gemm_workspace_.problem_count
    );

    gemm_workspace_.Reference->copy_from_device(gemm_workspace_.C->data());

    gemm_workspace_.arguments.batch_stride_A = gemm_workspace_.A->batch_stride();
    gemm_workspace_.arguments.batch_stride_B = gemm_workspace_.B->batch_stride();
    gemm_workspace_.arguments.batch_stride_C = gemm_workspace_.C->batch_stride();
    gemm_workspace_.arguments.batch_stride_D = gemm_workspace_.Computed->batch_stride();
  }

  //
  // Initialize the CUTLASS operation
  //
  Status status = Status::kSuccess;

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    if (options.execution_mode != ExecutionMode::kDryRun) {

      uint64_t workspace_size = underlying_operation->get_host_workspace_size(&gemm_workspace_.configuration);
      gemm_workspace_.host_workspace.resize(workspace_size, 0);

      workspace_size = underlying_operation->get_device_workspace_size(&gemm_workspace_.configuration,
                                                            &gemm_workspace_.arguments);
      gemm_workspace_.device_workspace.reset(library::NumericTypeID::kU8, workspace_size);

      status = underlying_operation->initialize(
        &gemm_workspace_.configuration,
        gemm_workspace_.host_workspace.data(),
        gemm_workspace_.device_workspace.data());

      if (status != Status::kSuccess) {
        return status;
      }

      if (problem_.split_k_mode == library::SplitKMode::kParallel) {
        workspace_size = reduction_op_->get_host_workspace_size(&gemm_workspace_.reduction_configuration);
        gemm_workspace_.reduction_host_workspace.resize(workspace_size, 0);

        status = reduction_op_->initialize(
          &gemm_workspace_.reduction_configuration,
          gemm_workspace_.reduction_host_workspace.data(),
          nullptr);

        if (status != Status::kSuccess) {
          return status;
        }
      }
    }

    //
    // If CUTLASS is enabled, generate a result for it
    //
    results_.push_back(model_result_);
    results_.back().provider = library::Provider::kCUTLASS;
    results_.back().op_kind = library::OperationKind::kGemm;
    results_.back().disposition = Disposition::kNotRun;

    for(auto provider : verification_providers_) {
      results_.back().verification_map[provider] = Disposition::kNotRun;
    }
  }

  return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies CUTLASS against references
bool GemmOperationProfiler::verify_cutlass(
  Options const &options,  
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (!options.profiling.provider_enabled(library::Provider::kCUTLASS)) {
    return true;
  }

  if (options.execution_mode == ExecutionMode::kDryRun) {
    return true;
  }

  // Initialize structure containing GEMM arguments
  gemm_workspace_.arguments.A = gemm_workspace_.A->data();
  gemm_workspace_.arguments.B = gemm_workspace_.B->data();
  gemm_workspace_.arguments.C = gemm_workspace_.C->data();
  gemm_workspace_.arguments.D = gemm_workspace_.Computed->data();
  gemm_workspace_.arguments.alpha = problem_.alpha.data();
  gemm_workspace_.arguments.beta = problem_.beta.data();
  gemm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;
  gemm_workspace_.arguments.batch_stride_A = gemm_workspace_.A->batch_stride();
  gemm_workspace_.arguments.batch_stride_B = gemm_workspace_.B->batch_stride();
  gemm_workspace_.arguments.batch_stride_C = gemm_workspace_.C->batch_stride();
  gemm_workspace_.arguments.batch_stride_D = gemm_workspace_.Computed->batch_stride();

  if (problem_.split_k_mode == library::SplitKMode::kParallel) {
    gemm_workspace_.arguments.D                       = gemm_workspace_.device_workspace.data();
    gemm_workspace_.arguments.alpha                   = problem_.alpha_one.data();
    gemm_workspace_.arguments.beta                    = problem_.beta_zero.data();

    gemm_workspace_.reduction_arguments.workspace     = gemm_workspace_.device_workspace.data();
    gemm_workspace_.reduction_arguments.source        = gemm_workspace_.C->data();
    gemm_workspace_.reduction_arguments.destination   = gemm_workspace_.Computed->data();
    gemm_workspace_.reduction_arguments.alpha         = problem_.alpha.data();
    gemm_workspace_.reduction_arguments.beta          = problem_.beta.data();
    gemm_workspace_.reduction_arguments.pointer_mode  = library::ScalarPointerMode::kHost;
  }

  //
  // Run the CUTLASS operation
  //

 // initialize gemm underlying operation to handle parallel reduction
  library::Operation const * underlying_operation = operation;

  if (problem_.split_k_mode == library::SplitKMode::kParallel) {
    if (!(underlying_operation = library::find_gemm_operation_for_parallel_reduction(operation))) {
      results_.back().disposition = Disposition::kFailed;
      return false;
    }
  }

  results_.back().status = underlying_operation->run(
    &gemm_workspace_.arguments, 
    gemm_workspace_.host_workspace.data(),
    gemm_workspace_.device_workspace.data());

  if (results_.back().status != Status::kSuccess) {
    results_.back().disposition = Disposition::kFailed;
    return false;
  }

  // Run parallel reduction kernel for parallel split_k_mode
  if (problem_.split_k_mode == library::SplitKMode::kParallel) {
    results_.back().status = reduction_op_->run(
      &gemm_workspace_.reduction_arguments,
      gemm_workspace_.reduction_host_workspace.data(),
      nullptr);

    if (results_.back().status != Status::kSuccess) {
      results_.back().disposition = Disposition::kFailed;
      return false;
    }
  }

  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    results_.back().disposition = Disposition::kFailed;
    return false;
  }

  // CUTLASS op ran the but not yet verified against any verification provider
  results_.back().disposition = Disposition::kNotVerified;

  //
  // Run verification providers
  //

  if (options.verification.enabled) {

#if CUTLASS_ENABLE_CUBLAS
    if (options.verification.provider_enabled(library::Provider::kCUBLAS)) {

      // Guard against unsupported cases
      auto const & gemm_desc = static_cast<library::GemmDescription const &>(operation->description());

      if (cublas_satisfies(gemm_desc) == Status::kSuccess) {

        // call cublas verification if supported
        verify_with_cublas_(
          options,
          report,
          device_context,
          operation,
          problem_space,
          problem);
        }

      else {
        // set verification map for cublas to not supported
        results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kNotSupported;
      }
    }
#endif // #if CUTLASS_ENABLE_CUBLAS

    verify_with_reference_(options, report, device_context, operation, problem_space, problem);
    
    // Update disposition to worst case verification outcome among all 
    // verification providers which are supported
    bool is_any_verification_run_passed = false;
    for(auto &m : results_.back().verification_map) {
      if(m.second == Disposition::kFailed || m.second == Disposition::kIncorrect) {
        results_.back().disposition = m.second;
        return true;
      }
      if(!is_any_verification_run_passed && m.second == Disposition::kPassed) {
        is_any_verification_run_passed = true;
      }
    }

    if(is_any_verification_run_passed) {
      results_.back().disposition = Disposition::kPassed;
    }
  }

  // Return true means continue profiling
  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies CUTLASS against references
bool GemmOperationProfiler::verify_with_cublas_(
  Options const &options,  
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {


#if CUTLASS_ENABLE_CUBLAS

  library::GemmDescription const &gemm_desc = 
    static_cast<library::GemmDescription const &>(operation->description());

  //
  // Construct cuBLAS operators
  //
    
  CublasCreate handle;
  cublasStatus_t status = handle.get_cublas_create_status();

  if (status != CUBLAS_STATUS_SUCCESS) {

    results_.back().verification_map[library::Provider::kCUBLAS] = get_cutlass_disposition(status);
    return true;
  }

  std::vector<cublasGemmAlgo_t> algorithms;

  detail::select_cublas_algorithms(
    algorithms, 
    options, 
    gemm_desc);

  if (algorithms.empty()) {
    // no algorithm selected
    return true;
  }

  //
  // Initialize state
  //

  try {

    //
    // Construct dispatcher to cublasGemmEx()
    //

    // Initialize structure containing GEMM arguments
    gemm_workspace_.arguments.A = gemm_workspace_.A->data();
    gemm_workspace_.arguments.batch_stride_A = gemm_workspace_.A->batch_stride();
    gemm_workspace_.arguments.B = gemm_workspace_.B->data();
    gemm_workspace_.arguments.batch_stride_B = gemm_workspace_.B->batch_stride();
    gemm_workspace_.arguments.C = gemm_workspace_.Reference->data();
    gemm_workspace_.arguments.batch_stride_C = gemm_workspace_.Reference->batch_stride();
    gemm_workspace_.arguments.D = gemm_workspace_.Reference->data();
    gemm_workspace_.arguments.batch_stride_D = gemm_workspace_.Reference->batch_stride();
    gemm_workspace_.arguments.alpha = problem_.alpha.data();
    gemm_workspace_.arguments.beta = problem_.beta.data();
    gemm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

    detail::cublasGemmExDispatcher gemm_op( 
      gemm_desc, 
      gemm_workspace_.configuration,
      gemm_workspace_.arguments,
      algorithms.front()
    );

    if (gemm_op.status != Status::kSuccess) {
      results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kNotRun;
      return true;
    }

    results_.back().status = Status::kSuccess;

    status = gemm_op(handle);

    // Handle errors
    if (status != CUBLAS_STATUS_SUCCESS) {

      results_.back().verification_map[library::Provider::kCUBLAS] = get_cutlass_disposition(status);
      return true;
    }

    //
    // Verify results
    //

    results_.back().verification_map[library::Provider::kCUBLAS] = compare_tensors(
      options,
      *gemm_workspace_.Computed,
      *gemm_workspace_.Reference,
      gemm_workspace_.Computed->batch_stride()
    );

    // Save workspace if incorrect
    if (options.verification.save_workspace == SaveWorkspace::kIncorrect && 
      results_.back().verification_map[library::Provider::kCUBLAS] == Disposition::kIncorrect) {

      save_workspace(
        device_context,
        options,
        gemm_desc,
        library::Provider::kCUTLASS,
        library::Provider::kCUBLAS);
    }
  }
  catch (...) {
    results_.back().verification_map[library::Provider::kCUBLAS] = Disposition::kFailed;
  }

#endif

  // Return true means continue profiling
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies CUTLASS against host and device references
bool GemmOperationProfiler::verify_with_reference_(
  Options const &options,  
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::GemmDescription const &gemm_desc = 
    static_cast<library::GemmDescription const &>(operation->description());

  //
  // Initialize state
  //

  library::Provider references[] = {
    library::Provider::kReferenceDevice,
    library::Provider::kReferenceHost
  };

  for (auto provider : references) {

    // Skip providers that are not enabled
    if (!options.verification.provider_enabled(provider)) {
      continue;
    }

    void *ptr_A = gemm_workspace_.A->data();
    void *ptr_B = gemm_workspace_.B->data();
    void *ptr_C = gemm_workspace_.C->data();
    void *ptr_D = gemm_workspace_.Reference->data();

    // To support the host-side reference, conditionally allocate and
    // copy tensors to host memory.
    std::vector<uint8_t> host_data_A;
    std::vector<uint8_t> host_data_B;
    std::vector<uint8_t> host_data_C;
    std::vector<uint8_t> host_data_D;

    if (provider == library::Provider::kReferenceHost) {

      host_data_A.resize(gemm_workspace_.A->bytes());
      ptr_A = host_data_A.data();
      gemm_workspace_.A->copy_to_host(ptr_A);

      host_data_B.resize(gemm_workspace_.B->bytes());
      ptr_B = host_data_B.data();
      gemm_workspace_.B->copy_to_host(ptr_B);

      host_data_C.resize(gemm_workspace_.C->bytes());
      ptr_C = host_data_C.data();
      gemm_workspace_.C->copy_to_host(ptr_C);

      host_data_D.resize(gemm_workspace_.Reference->bytes());
      ptr_D = host_data_D.data();
    }

    //
    // Launch
    //

    library::Handle handle;

    handle.set_provider(provider);

    Status status = handle.gemm_universal(
      problem_.mode,
      gemm_workspace_.configuration.problem_size.m(),
      gemm_workspace_.configuration.problem_size.n(),
      gemm_workspace_.configuration.problem_size.k(),
      gemm_desc.tile_description.math_instruction.element_accumulator,
      gemm_desc.element_epilogue,

      problem_.alpha.data(),

      gemm_desc.A.element,
      gemm_desc.A.layout,
      gemm_desc.transform_A,
      ptr_A,
      int(gemm_workspace_.configuration.lda),

      gemm_desc.B.element,
      gemm_desc.B.layout,
      gemm_desc.transform_B,
      ptr_B,
      int(gemm_workspace_.configuration.ldb),

      problem_.beta.data(),

      gemm_desc.C.element,
      ptr_C,
      int(gemm_workspace_.configuration.ldc),

      ptr_D,
      int(gemm_workspace_.configuration.ldd),

      gemm_workspace_.configuration.batch_count,
      gemm_workspace_.A->batch_stride(),
      gemm_workspace_.B->batch_stride(),
      gemm_workspace_.C->batch_stride(),
      gemm_workspace_.Reference->batch_stride()
    );

    if (status != Status::kSuccess) {
      results_.back().verification_map[provider] = Disposition::kNotRun;
      return true;
    }

    results_.back().status = status;

    if (provider == library::Provider::kReferenceHost) {
      gemm_workspace_.Reference->copy_from_host(ptr_D); 
    }

    //
    // Verify results
    //

    results_.back().verification_map[provider] = compare_tensors(
      options,
      *gemm_workspace_.Computed,
      *gemm_workspace_.Reference,
      gemm_workspace_.Computed->batch_stride()
    );

    // Save workspace if incorrect
    if (options.verification.save_workspace == SaveWorkspace::kIncorrect && 
      results_.back().verification_map[provider] == Disposition::kIncorrect) {

      save_workspace(
        device_context,
        options,
        gemm_desc,
        library::Provider::kCUTLASS,
        provider);
    }
  }

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Measures performance results
bool GemmOperationProfiler::profile(
  Options const &options,  
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    // Initialize structure containing GEMM arguments
    gemm_workspace_.arguments.A = gemm_workspace_.A->data();
    gemm_workspace_.arguments.B = gemm_workspace_.B->data();
    gemm_workspace_.arguments.C = gemm_workspace_.C->data();
    gemm_workspace_.arguments.D = gemm_workspace_.Computed->data();
    gemm_workspace_.arguments.alpha = problem_.alpha.data();
    gemm_workspace_.arguments.beta = problem_.beta.data();
    gemm_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;
    gemm_workspace_.arguments.batch_stride_A = gemm_workspace_.A->batch_stride();
    gemm_workspace_.arguments.batch_stride_B = gemm_workspace_.B->batch_stride();
    gemm_workspace_.arguments.batch_stride_C = gemm_workspace_.C->batch_stride();
    gemm_workspace_.arguments.batch_stride_D = gemm_workspace_.Computed->batch_stride();

    if (problem_.split_k_mode == library::SplitKMode::kParallel) {
      gemm_workspace_.arguments.D                       = gemm_workspace_.device_workspace.data();
      gemm_workspace_.arguments.alpha                   = problem_.alpha_one.data();
      gemm_workspace_.arguments.beta                    = problem_.beta_zero.data();

      gemm_workspace_.reduction_arguments.workspace     = gemm_workspace_.device_workspace.data();
      gemm_workspace_.reduction_arguments.source        = gemm_workspace_.C->data();
      gemm_workspace_.reduction_arguments.destination   = gemm_workspace_.Computed->data();
      gemm_workspace_.reduction_arguments.alpha         = problem_.alpha.data();
      gemm_workspace_.reduction_arguments.beta          = problem_.beta.data();
      gemm_workspace_.reduction_arguments.pointer_mode  = library::ScalarPointerMode::kHost;
    }

    results_.back().status = profile_cutlass_(
      results_.back().runtime,
      options,
      operation,
      &gemm_workspace_.arguments,
      gemm_workspace_.host_workspace.data(),
      gemm_workspace_.device_workspace.data()
    );
  }
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Method to profile a CUTLASS Operation
Status GemmOperationProfiler::profile_cutlass_(
  double &runtime,
  Options const &options,
  library::Operation const *operation,
  void *arguments,
  void *host_workspace,
  void *device_workspace) {

  GpuTimer timer;

  // initialize gemm underlying operation to handle parallel reduction
  library::Operation const * underlying_operation = operation;

  if (problem_.split_k_mode == library::SplitKMode::kParallel) {
    if (!(underlying_operation = library::find_gemm_operation_for_parallel_reduction(operation))) {
      return Status::kErrorNotSupported;
    }
  }

  //
  // Optional sleep to limit power consumption and thermals
  //

  sleep(options.profiling.sleep_duration);

  //
  // Warmup loop
  //

  Status status;

  for (int iteration = 0; iteration < options.profiling.warmup_iterations; ++iteration) {
    
    int problem_idx = (iteration % gemm_workspace_.problem_count) * problem_.batch_count;

    gemm_workspace_.arguments.A = gemm_workspace_.A->batch_data(problem_idx);
    gemm_workspace_.arguments.B = gemm_workspace_.B->batch_data(problem_idx);
    gemm_workspace_.arguments.C = gemm_workspace_.C->batch_data(problem_idx);
    gemm_workspace_.arguments.D = gemm_workspace_.Computed->batch_data(problem_idx);

    if (problem_.split_k_mode == library::SplitKMode::kParallel) {
      gemm_workspace_.arguments.D                     = gemm_workspace_.device_workspace.data();

      gemm_workspace_.reduction_arguments.workspace   = gemm_workspace_.device_workspace.data();
      gemm_workspace_.reduction_arguments.source      = gemm_workspace_.C->batch_data(problem_idx);
      gemm_workspace_.reduction_arguments.destination = gemm_workspace_.Computed->batch_data(problem_idx);
    }

    // Execute the CUTLASS operation
    status = underlying_operation->run(
      &gemm_workspace_.arguments,
      host_workspace,
      device_workspace);

    if (status != Status::kSuccess) {
      return status;
    }

    // Run parallel reduction kernel for parallel split_k_mode
    if (problem_.split_k_mode == library::SplitKMode::kParallel) {
      status = reduction_op_->run(
        &gemm_workspace_.reduction_arguments,
        gemm_workspace_.reduction_host_workspace.data(),
        nullptr);

      if (status != Status::kSuccess) {
        return status;
      }
    }
  }

  //
  // Initialize GPU timer
  //

  timer.start();

  //
  // Profiling loop
  //

  int Iterations = options.profiling.iterations;

  int iteration = 0;
  for (; iteration < Iterations; ++iteration) {
    
    // Iterate over copies of the problem in memory
    int workspace_idx = options.profiling.warmup_iterations + iteration;
    int problem_idx = (workspace_idx % gemm_workspace_.problem_count) * problem_.batch_count;

    gemm_workspace_.arguments.A = gemm_workspace_.A->batch_data(problem_idx);
    gemm_workspace_.arguments.B = gemm_workspace_.B->batch_data(problem_idx);
    gemm_workspace_.arguments.C = gemm_workspace_.C->batch_data(problem_idx);
    gemm_workspace_.arguments.D = gemm_workspace_.Computed->batch_data(problem_idx);

    if (problem_.split_k_mode == library::SplitKMode::kParallel) {
      gemm_workspace_.arguments.D                     = gemm_workspace_.device_workspace.data();

      gemm_workspace_.reduction_arguments.workspace   = gemm_workspace_.device_workspace.data();
      gemm_workspace_.reduction_arguments.source      = gemm_workspace_.C->batch_data(problem_idx);
      gemm_workspace_.reduction_arguments.destination = gemm_workspace_.Computed->batch_data(problem_idx);
    }

    status = underlying_operation->run(
      arguments,
      host_workspace,
      device_workspace);

    if (status != Status::kSuccess) {
      return status;
    }

    // Run parallel reduction kernel for parallel split_k_mode
    if (problem_.split_k_mode == library::SplitKMode::kParallel) {
      status = reduction_op_->run(
        &gemm_workspace_.reduction_arguments,
        gemm_workspace_.reduction_host_workspace.data(),
        nullptr);

      if (status != Status::kSuccess) {
        return status;
      }
    }
  }

  //
  // Wait for completion
  //

  timer.stop_and_wait();

  //
  // Update performance result
  //

  runtime = timer.duration(iteration);

  return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
