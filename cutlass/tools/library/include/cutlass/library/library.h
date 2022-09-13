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
/*! 
  \file

  \brief CUTLASS Library is an object-oriented approach to managing operations implemented by CUTLASS.

  Generally,
    
    description   - compile-time constant parameters used to instantiate an operation

    configuration - runtime parameters with computationally expensive initialization 
    
    arguments     - runtime parameters that may be passed to an initialized operation with low
                    computational overhead
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/blas3.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Layout type identifier
enum class LayoutTypeID {
  kUnknown,
  kColumnMajor,
  kRowMajor,
  kColumnMajorInterleavedK2,
  kRowMajorInterleavedK2,
  kColumnMajorInterleavedK4,
  kRowMajorInterleavedK4,
  kColumnMajorInterleavedK16,
  kRowMajorInterleavedK16,
  kColumnMajorInterleavedK32,
  kRowMajorInterleavedK32,
  kColumnMajorInterleavedK64,
  kRowMajorInterleavedK64,
  kTensorNCHW,
  kTensorNCDHW,
  kTensorNHWC,
  kTensorNDHWC,
  kTensorNC32HW32,
  kTensorC32RSK32,
  kTensorNC64HW64,
  kTensorC64RSK64,
  kInvalid
};
  
/// Numeric data type
enum class NumericTypeID {
  kUnknown,
  kVoid,
  kB1,
  kU2,
  kU4,
  kU8,
  kU16,
  kU32,
  kU64,
  kS2,
  kS4,
  kS8,
  kS16,
  kS32,
  kS64,
  kF16,
  kBF16, 
  kTF32,
  kF32,
  kF64,
  kCF16,
  kCBF16,
  kCF32,
  kCTF32,
  kCF64,
  kCS2,
  kCS4,
  kCS8,
  kCS16,
  kCS32,
  kCS64,
  kCU2,
  kCU4,
  kCU8,
  kCU16,
  kCU32,
  kCU64,
  kInvalid
};

/// Enumerated type describing a transformation on a complex value.
enum class ComplexTransform {
  kNone,
  kConjugate,
  kInvalid
};

/// Providers
enum class Provider {
  kNone,
  kCUTLASS,
  kReferenceHost,
  kReferenceDevice,
  kCUBLAS,
  kCUDNN,               
  kInvalid
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumeration indicating the kind of operation
enum class OperationKind {
  kGemm,
  kRankK,
  kRank2K,
  kTrmm,
  kSymm,
  kConv2d,              
  kConv3d,             
  kEqGemm,
  kSparseGemm,
  kReduction,
  kInvalid
};

/// Enumeration indicating whether scalars are in host or device memory
enum class ScalarPointerMode {
  kHost,
  kDevice,
  kInvalid
};

/// Describes how reductions are performed across threadblocks
enum class SplitKMode {
  kNone,
  kSerial,
  kParallel,
  kParallelSerial,
  kInvalid
};

/// Indicates the classificaition of the math instruction
enum class OpcodeClassID {
  kSimt,
  kTensorOp,
  kWmmaTensorOp,
  kSparseTensorOp,
  kInvalid
};

enum class MathOperationID {
  kAdd,
  kMultiplyAdd,
  kMultiplyAddSaturate,
  kMultiplyAddFastBF16,
  kMultiplyAddFastF16,
  kMultiplyAddFastF32,              
  kMultiplyAddComplex,
  kMultiplyAddComplexFastF32,      
  kMultiplyAddGaussianComplex,
  kXorPopc,
  kInvalid
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumeration indicating what kind of GEMM operation to perform
enum class GemmKind {
  kGemm,
  kSparse,
  kUniversal,
  kPlanarComplex,
  kPlanarComplexArray,
  kGrouped,
  kInvalid
};

/// Mode of Universal GEMM
using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

/// Enumeration indicating what kind of RankK update operation to perform
enum class RankKKind {
  kUniversal,
  kInvalid
};

/// Enumeration indicating what kind of TRMM operation to perform
enum class TrmmKind {
  kUniversal,
  kInvalid
};

/// Enumeration indicating what kind of SYMM/HEMM operation to perform
enum class SymmKind {
  kUniversal,
  kInvalid
};

/// Enumeration indicating what kind of Conv2d operation to perform
enum class ConvKind {
  kUnknown,
  kFprop,
  kDgrad,
  kWgrad,
  kInvalid
};

enum class ConvModeID {
  kCrossCorrelation,
  kConvolution,
  kInvalid
};

// Iterator algorithm enum in order of general performance-efficiency
enum class IteratorAlgorithmID {
  kNone,
  kAnalytic,
  kOptimized,
  kFixedChannels,
  kFewChannels,
  kInvalid
};


enum class EpilogueKind {
  kUnknown,
  kConversion,
  kLinearCombination,
  kLinearCombinationClamp,
  kLinearCombinationPlanarComplex,
  kLinearCombinationRelu,
  kLinearCombinationSigmoid,
  kInvalid
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct MathInstructionDescription {

  /// Shape of the target math instruction
  cutlass::gemm::GemmCoord instruction_shape;

  /// Describes the data type of the internal accumulator
  NumericTypeID element_accumulator;

  /// Classification of math instruction
  OpcodeClassID opcode_class;

  /// Type of math operation performed
  MathOperationID math_operation;

  //
  // Methods
  //

  MathInstructionDescription(
    cutlass::gemm::GemmCoord instruction_shape = cutlass::gemm::GemmCoord(),
    NumericTypeID element_accumulator = NumericTypeID::kInvalid,
    OpcodeClassID opcode_class = OpcodeClassID::kInvalid,
    MathOperationID math_operation = MathOperationID::kMultiplyAdd
  ):
    instruction_shape(instruction_shape), 
    element_accumulator(element_accumulator), 
    opcode_class(opcode_class),
    math_operation(math_operation) {}

  // Equality operator
  inline
  bool operator==(MathInstructionDescription const& rhs) const{
    return (
      (instruction_shape == rhs.instruction_shape) &&
      (element_accumulator == rhs.element_accumulator) &&
      (opcode_class == rhs.opcode_class) &&
      (math_operation == rhs.math_operation));
  }

  // Inequality operator
  inline
  bool operator!=(MathInstructionDescription const& rhs) const {
    return !(*this == rhs);
  }

};

/// Structure describing the tiled structure of a GEMM-like computation
struct TileDescription {

  /// Describes the shape of a threadblock (in elements)
  cutlass::gemm::GemmCoord threadblock_shape;

  /// Describes the number of pipeline stages in the threadblock-scoped mainloop
  int threadblock_stages;

  /// Number of warps in each logical dimension
  cutlass::gemm::GemmCoord warp_count;

  /// Core math instruction
  MathInstructionDescription math_instruction;

  /// Minimum compute capability (e.g. 70, 75) of a device eligible to run the operation.
  int minimum_compute_capability;

  /// Minimum compute capability (e.g. 70, 75) of a device eligible to run the operation.
  int maximum_compute_capability;

  //
  // Methods
  //

  TileDescription(
    cutlass::gemm::GemmCoord threadblock_shape = cutlass::gemm::GemmCoord(),
    int threadblock_stages = 0,
    cutlass::gemm::GemmCoord warp_count = cutlass::gemm::GemmCoord(),
    MathInstructionDescription math_instruction = MathInstructionDescription(),
    int minimum_compute_capability = 0,
    int maximum_compute_capability = 0
  ):
    threadblock_shape(threadblock_shape), 
    threadblock_stages(threadblock_stages), 
    warp_count(warp_count),
    math_instruction(math_instruction),
    minimum_compute_capability(minimum_compute_capability),
    maximum_compute_capability(maximum_compute_capability) { }

  // Equality operator
  inline
  bool operator==(TileDescription const& rhs) const{
    return (
      (threadblock_shape == rhs.threadblock_shape) &&
      (threadblock_stages == rhs.threadblock_stages) &&
      (warp_count == rhs.warp_count) &&
      (math_instruction == rhs.math_instruction) &&
      (minimum_compute_capability == rhs.minimum_compute_capability) &&
      (maximum_compute_capability == rhs.maximum_compute_capability));
  }

  // Inequality operator
  inline
  bool operator!=(TileDescription const& rhs) const {
    return !(*this == rhs);
  }
};

/// High-level description of an operation
struct OperationDescription {

  /// Unique identifier describing the operation
  char const * name;

  /// Operation provider
  Provider provider;

  /// Kind of operation
  OperationKind kind;

  /// Describes the tiled structure of a GEMM-like computation
  TileDescription tile_description;

  //
  // Methods
  //
  OperationDescription(
    char const * name = "unknown",
    Provider Provider = Provider::kInvalid,
    OperationKind kind = OperationKind::kInvalid, 
    TileDescription const & tile_description = TileDescription()
  ):
    name(name), kind(kind), tile_description(tile_description) { }
};

/// Structure describing the properties of a tensor
struct TensorDescription {

  /// Numeric type of an individual element
  NumericTypeID element;

  /// Enumerant identifying the layout function for the tensor
  LayoutTypeID layout;

  /// Alignment restriction on pointers, strides, and extents
  int alignment;

  /// log2() of the maximum extent of each dimension
  int log_extent_range;

  /// log2() of the maximum value each relevant stride may have
  int log_stride_range;
  
  //
  // Methods
  //

  TensorDescription(
    NumericTypeID element = NumericTypeID::kInvalid,
    LayoutTypeID layout = LayoutTypeID::kInvalid,
    int alignment = 1,
    int log_extent_range = 24,
    int log_stride_range = 24
  ):
    element(element), 
    layout(layout), 
    alignment(alignment), 
    log_extent_range(log_extent_range), 
    log_stride_range(log_stride_range)  { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Description of all GEMM computations
struct GemmDescription : public OperationDescription {

  /// Indicates the kind of GEMM performed
  GemmKind gemm_kind;
  
  /// Describes the A operand
  TensorDescription A;

  /// Describes the B operand
  TensorDescription B;

  /// Describes the source and destination matrices
  TensorDescription C;

  /// Describes the sparse meta matrices
  TensorDescription E;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  /// Describes the structure of parallel reductions
  SplitKMode split_k_mode;

  /// Transformation on A operand
  ComplexTransform transform_A;

  /// Transformation on B operand
  ComplexTransform transform_B;

  //
  // Methods
  //

  GemmDescription(
    GemmKind gemm_kind = GemmKind::kGemm,
    TensorDescription const &A = TensorDescription(),
    TensorDescription const &B = TensorDescription(),
    TensorDescription const &C = TensorDescription(),
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone,
    ComplexTransform transform_B = ComplexTransform::kNone
  ):
    gemm_kind(gemm_kind),
    A(A),
    B(B),
    C(C),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {} 
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Desciprion for structured sparse GEMMs.
struct SparseGemmDescription : public GemmDescription {

  /// Description structure for structured sparse GEMM
  SparseGemmDescription(
    GemmKind gemm_kind = GemmKind::kGemm,
    TensorDescription const &A = TensorDescription(),
    TensorDescription const &B = TensorDescription(),
    TensorDescription const &C = TensorDescription(),
    TensorDescription const &E = TensorDescription(),
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone,
    ComplexTransform transform_B = ComplexTransform::kNone
  ):
    GemmDescription(gemm_kind, A, B, C, element_epilogue, split_k_mode, transform_A, transform_B)
     {this->E = E;}
};

/// Description of all Reduction operations
struct ReductionDescription : public OperationDescription {

  /// Describes the data type of workspace
  NumericTypeID element_workspace;

  /// Describes the data type of final output
  NumericTypeID element_output;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;
};

/// Description of all Rank K update computations (SYRK, HERK, SYR2K, HER2K)
struct RankKDescription : public OperationDescription {

  /// Indicates which device template is used (universal or regular)
  RankKKind rank_k_kind;

  /// Number of rank update (rank k or rank 2k)
  int num_ranks;
  
  /// Describes the A operand
  TensorDescription A;

  /// Describes the B operand (used only for SYR2K and HER2K)
  TensorDescription B;

  /// Describes the source and destination matrices
  TensorDescription C;

  /// Describes the fill mode for matrix C
  FillMode fill_mode;

  /// Describes the blas mode (symmetric/hermitian)
  BlasMode blas_mode;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  /// Describes the structure of parallel reductions
  SplitKMode split_k_mode;

  /// Transformation on A operand
  ComplexTransform transform_A;

  /// Transformation on B operand
  ComplexTransform transform_B;

  //
  // Methods
  //

  RankKDescription(
    RankKKind rank_k_kind = RankKKind::kUniversal,
    int num_ranks = 1,
    TensorDescription const &A = TensorDescription(),
    TensorDescription const &B = TensorDescription(),
    TensorDescription const &C = TensorDescription(),
    FillMode fill_mode = FillMode::kInvalid,
    BlasMode blas_mode = BlasMode::kInvalid,
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone,
    ComplexTransform transform_B = ComplexTransform::kNone
  ):
    rank_k_kind(rank_k_kind),
    num_ranks(num_ranks),
    A(A),
    B(B),
    C(C),
    fill_mode(fill_mode),
    blas_mode(blas_mode),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {} 
};
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Description of all TRMM computations
struct TrmmDescription : public OperationDescription {

  /// Indicates the kind of TRMM performed
  TrmmKind trmm_kind;
  
  /// Describes the A operand
  TensorDescription A;

  /// Describes the side mode for matrix A
  SideMode side_mode;

  /// Describes the fill mode for matrix A
  FillMode fill_mode;

  /// Describes the diag type for matrix A
  DiagType diag_type;

  /// Describes the B operand
  TensorDescription B;

  /// Describes the source and destination matrices
  TensorDescription D;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  /// Describes the structure of parallel reductions
  SplitKMode split_k_mode;

  /// Transformation on A operand
  ComplexTransform transform_A;

  //
  // Methods
  //

  TrmmDescription(
    TrmmKind trmm_kind = TrmmKind::kUniversal,
    TensorDescription const &A = TensorDescription(),
    SideMode side_mode = SideMode::kInvalid,
    FillMode fill_mode = FillMode::kInvalid,
    DiagType diag_type = DiagType::kInvalid,
    TensorDescription const &B = TensorDescription(),
    TensorDescription const &D = TensorDescription(),
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone
  ):
    trmm_kind(trmm_kind),
    A(A),
    side_mode(side_mode),
    fill_mode(fill_mode),
    diag_type(diag_type),
    B(B),
    D(D),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A) {} 
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Description of all SYMM/HEMM update computations
struct SymmDescription : public OperationDescription {

  /// Indicates which device template is used (universal or regular)
  SymmKind symm_kind;
  
  /// Describes the A operand
  TensorDescription A;

  /// Describes the B operand 
  TensorDescription B;

  /// Describes the source and destination matrices
  TensorDescription C;

  /// Describes the side mode for matrix A
  SideMode side_mode;

  /// Describes the fill mode for matrix A
  FillMode fill_mode;

  /// Describes the blas mode (symmetric/hermitian)
  BlasMode blas_mode;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  /// Describes the structure of parallel reductions
  SplitKMode split_k_mode;

  /// Transformation on A operand
  ComplexTransform transform_A;

  /// Transformation on B operand
  ComplexTransform transform_B;

  //
  // Methods
  //

  SymmDescription(
    SymmKind symm_kind = SymmKind::kUniversal,
    TensorDescription const &A = TensorDescription(),
    TensorDescription const &B = TensorDescription(),
    TensorDescription const &C = TensorDescription(),
    SideMode side_mode = SideMode::kInvalid,
    FillMode fill_mode = FillMode::kInvalid,
    BlasMode blas_mode = BlasMode::kInvalid,
    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
    SplitKMode split_k_mode = SplitKMode::kNone,
    ComplexTransform transform_A = ComplexTransform::kNone,
    ComplexTransform transform_B = ComplexTransform::kNone
  ):
    symm_kind(symm_kind),
    A(A),
    B(B),
    C(C),
    side_mode(side_mode),
    fill_mode(fill_mode),
    blas_mode(blas_mode),
    element_epilogue(element_epilogue),
    split_k_mode(split_k_mode),
    transform_A(transform_A),
    transform_B(transform_B) {} 
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Description of all Conv2d operations
struct ConvDescription : public OperationDescription {
  /// Describes the convolution dimension support (2D or 3D)
  int conv_dim;
  
  /// Describes the kind of convolution
  ConvKind conv_kind;

  /// Describes the type of iterator algorithm (analytic or precomputed)
  IteratorAlgorithmID iterator_algorithm;

  /// Describes the A operand
  TensorDescription A;

  /// Describes the B operand
  TensorDescription B;

  /// Describes the C operand
  TensorDescription C;

  /// Describes the data type of the scalars passed to the epilogue
  NumericTypeID element_epilogue;

  //
  // Methods
  //
  // Returns Activation TensorDescription
  TensorDescription activation() const {
    switch(conv_kind) {
      case library::ConvKind::kFprop : return A;
      case library::ConvKind::kDgrad : return C;
      case library::ConvKind::kWgrad : return B;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

  // Returns Filter TensorDescription
  TensorDescription filter() const {
    switch(conv_kind) {
      case library::ConvKind::kFprop : return B;
      case library::ConvKind::kDgrad : return B;
      case library::ConvKind::kWgrad : return C;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

  // Returns Output TensorDescription
  TensorDescription output() const {
    switch(conv_kind) {
      case library::ConvKind::kFprop : return C;
      case library::ConvKind::kDgrad : return A;
      case library::ConvKind::kWgrad : return A;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Base class for all operations
class Operation {
public:

  virtual ~Operation() { }

  virtual OperationDescription const & description() const = 0;

  virtual Status can_implement(
    void const *configuration, 
    void const *arguments) const = 0;
  
  virtual uint64_t get_host_workspace_size(
    void const *configuration) const = 0;
  
  virtual uint64_t get_device_workspace_size(
    void const *configuration,
    void const *arguments = nullptr) const = 0;
  
  virtual Status initialize(
    void const *configuration, 
    void *host_workspace, 
    void *device_workspace = nullptr, 
    cudaStream_t stream = nullptr) const = 0;

  virtual Status run(
    void const *arguments,
    void *host_workspace, 
    void *device_workspace = nullptr, 
    cudaStream_t stream = nullptr) const = 0;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for basic GEMM operations
//
// OperationKind: Gemm
// GemmKind:      Gemm
//
struct GemmConfiguration {

  /// GEMM problem size
  gemm::GemmCoord problem_size;

  /// Leading dimension of A matrix
  int64_t lda;

  /// Leading dimension of B matrix
  int64_t ldb;

  /// Leading dimension of C matrix
  int64_t ldc;

  /// Leading dimension of D matrix
  int64_t ldd;

  /// Number of partitions of K dimension
  int split_k_slices;
};

/// Arguments for GEMM
struct GemmArguments {

  /// Pointer to A matrix
  void const *A;

  /// Pointer to B matrix
  void const *B;

  /// Pointer to C matrix
  void const *C;

  /// Pointer to D matrix
  void *D;

  /// Host or device pointer to alpha scalar
  void const *alpha;

  /// Host or device pointer to beta scalar
  void const *beta;

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for batched GEMM in which multiple matrix products are computed
//
// OperationKind: Gemm
// GemmKind:      Batched

struct GemmBatchedConfiguration {

  /// GEMM problem size
  gemm::GemmCoord problem_size;

  /// Leading dimension of A matrix
  int64_t lda;

  /// Leading dimension of B matrix
  int64_t ldb;

  /// Leading dimension of C matrix
  int64_t ldc;

  /// Leading dimension of D matrix
  int64_t ldd;

  /// Stride between instances of the A matrix in memory
  int64_t batch_stride_A;

  /// Stride between instances of the B matrix in memory
  int64_t batch_stride_B;

  /// Stride between instances of the C matrix in memory
  int64_t batch_stride_C;

  /// Stride between instances of the D matrix in memory
  int64_t batch_stride_D;

  /// Number of GEMMs in batch
  int batch_count;
};

/// Arguments to batched GEMM
using GemmBatchedArguments = GemmArguments;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for batched GEMM in which multiple matrix products are computed
//
// OperationKind: Gemm
// GemmKind:      Array

struct GemmArrayConfiguration {

  gemm::GemmCoord problem_size;
  
  /// Leading dimension of A matrix
  int64_t lda;

  /// Leading dimension of B matrix
  int64_t ldb;

  /// Leading dimension of C matrix
  int64_t ldc;

  /// Leading dimension of D matrix
  int64_t ldd;

  int batch_count;
};

/// Arguments for GEMM - used by all the GEMM operations
struct GemmArrayArguments {
  void const * const *A;
  void const * const *B;
  void const * const *C;
  void * const *D;
  void const *alpha;
  void const *beta;
  ScalarPointerMode pointer_mode;  
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Universal GEMM supporting multiple split-K modes, multiple batched modes, real and complex
//
// OperationKind: Gemm
// GemmKind:      Universal

struct GemmUniversalConfiguration {

  GemmUniversalMode mode;
  gemm::GemmCoord problem_size;
  int batch_count;

  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int64_t ldd;
};

struct GemmUniversalArguments {

  void const *A;
  void const *B;
  void const *C;
  void *D;

  void const *alpha;
  void const *beta;
  ScalarPointerMode pointer_mode;

  int64_t batch_stride_A;
  int64_t batch_stride_B;
  int64_t batch_stride_C;
  int64_t batch_stride_D;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Complex valued GEMM in which real and imaginary parts are separated by a stride
//
// OperationKind: Gemm
// GemmKind:      Planar complex

struct GemmPlanarComplexConfiguration {

  GemmUniversalMode mode;
  gemm::GemmCoord problem_size;
  int batch_count;

  int64_t lda_real;
  int64_t lda_imag;

  int64_t ldb_real;
  int64_t ldb_imag;

  int64_t ldc_real;
  int64_t ldc_imag;

  int64_t ldd_real;
  int64_t ldd_imag;
};

/// Arguments for planar complex GEMMs
struct GemmPlanarComplexArguments {

  void const *A_real;
  void const *A_imag;

  void const *B_real;
  void const *B_imag;

  void const *C_real;
  void const *C_imag;

  void *D_real;
  void *D_imag;

  void const *alpha;
  void const *beta;
  ScalarPointerMode pointer_mode;

  int64_t batch_stride_A_real;
  int64_t batch_stride_A_imag;

  int64_t batch_stride_B_real;
  int64_t batch_stride_B_imag;

  int64_t batch_stride_C_real;
  int64_t batch_stride_C_imag;

  int64_t batch_stride_D_real;
  int64_t batch_stride_D_imag;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This is a special form of planar complex which loads pointers and problem size
/// from memory.
struct GemmPlanarComplexArrayConfiguration {

  gemm::GemmCoord problem_size;
  int batch_count;

  int64_t lda_real;
  int64_t lda_imag;

  int64_t ldb_real;
  int64_t ldb_imag;

  int64_t ldc_real;
  int64_t ldc_imag;

  int64_t ldd_real;
  int64_t ldd_imag;
};

/// Arguments for planar complex GEMMs
struct GemmPlanarComplexArrayArguments {

  int const *M;
  int const *N;
  int const *K;

  void const * const * A_real;
  void const * const * A_imag;
  void const * const * B_real;
  void const * const * B_imag;
  void const * const * C_real;
  void const * const * C_imag;
  void * const * D_real;
  void * const * D_imag;

  void const * alpha;
  void const * beta;
  ScalarPointerMode pointer_mode;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Grouped GEMM supporting
//
// OperationKind: Gemm
// GemmKind:      Grouped

struct GemmGroupedConfiguration {

  int problem_count;
  int threadblock_count;

};

struct GemmGroupedArguments {

  gemm::GemmCoord *problem_sizes;

  void * ptr_A;
  void * ptr_B;
  void * ptr_C;
  void * ptr_D;

  int64_t *lda;
  int64_t *ldb;
  int64_t *ldc;
  int64_t *ldd;

  void const *alpha;
  void const *beta;
  ScalarPointerMode pointer_mode;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// OperationKind: kSparseGemm
//

/// Computes GEMM assumine one of the inputs has 2:4 structured sparsity.
struct SparseGemmConfiguration {

  GemmUniversalMode mode;
  gemm::GemmCoord problem_size;
  int batch_count;                /// number of sparse matrix products in batch

  int64_t lda;                    /// leading dimension of A operand
  int64_t ldb;                    /// leading dimension of B operand
  int64_t ldc;                    /// leading dimension of C operand
  int64_t ldd;                    /// leading dimension of D operand
  int64_t lde;                    /// leading dimension of E operand (metadata matrix)

  int64_t batch_stride_A;         // stride between matrices
  int64_t batch_stride_B;         // stride between matrices
  int64_t batch_stride_C;         // stride between matrices
  int64_t batch_stride_D;         // stride between matrices
  int64_t batch_stride_E;         // stride between matrices
};

/// Arguments for sparse GEMMs
struct SparseGemmArguments {

  void const *A;                    /// pointer to A matrix
  void const *B;                    /// pointer to B matrix
  void const *C;                    /// pointer to C matrix
  void *D;                          /// pointer to D matrix
  void const *E;                    /// pointer to E matric (metadata)

  void const *alpha;                /// pointer to alpha scalar
  void const *beta;                 /// pointer to beta scalar
  ScalarPointerMode pointer_mode;   /// enumerant indicating whether alpha/beta pointers are host
                                    ///   or device pointers.
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for basic Rank K update operations
//
// OperationKind: (Syrk, Herk, Syr2k, Her2k)
// RankKKind:      Universal
//
struct RankKConfiguration {

  /// SYRK problem size
  gemm::GemmCoord problem_size;

  /// Leading dimension of A matrix
  int64_t lda;

  /// Leading dimension of B matrix
  int64_t ldb;

  /// Leading dimension of C matrix
  int64_t ldc;

  /// Leading dimension of D matrix
  int64_t ldd;

  /// Batch Count
  int batch_count;
};

/// Arguments for (Syrk, Herk, Syr2k, Her2k)
struct RankKArguments {

  /// Pointer to A matrix
  void const *A;

  /// Pointer to B matrix (used only for Syr2k and Her2k)
  void const *B;

  /// Pointer to C matrix
  void const *C;

  /// Pointer to D matrix
  void *D;

  /// Host or device pointer to alpha scalar
  void const *alpha;

  /// Host or device pointer to beta scalar
  void const *beta;

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode;

  int64_t batch_stride_A;
  int64_t batch_stride_B;
  int64_t batch_stride_C;
  int64_t batch_stride_D;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for basic TRMM operations
//
// OperationKind: Trmm
// TrmmKind:      Universal
//
struct TrmmConfiguration {

  /// TRMM problem size
  gemm::GemmCoord problem_size;

  /// Leading dimension of A matrix
  int64_t lda;

  /// Leading dimension of B matrix
  int64_t ldb;

  /// Leading dimension of D matrix
  int64_t ldd;

  /// Batch Count
  int batch_count;
};

/// Arguments for TRMM
struct TrmmArguments {

  /// Pointer to A matrix
  void const *A;

  /// Pointer to B matrix
  void const *B;

  /// Pointer to D matrix
  void *D;

  /// Host or device pointer to alpha scalar
  void const *alpha;

  /// Host or device pointer to beta scalar
  void const *beta;

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode;

  int64_t batch_stride_A;
  int64_t batch_stride_B;
  int64_t batch_stride_D;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for basic SYMM/HEMM update operations
//
// OperationKind: (Symm, Hemm)
// SymmKind:      Universal
//
struct SymmConfiguration {

  /// SYMM/HEMM problem size
  gemm::GemmCoord problem_size;

  /// Leading dimension of A matrix
  int64_t lda;

  /// Leading dimension of B matrix
  int64_t ldb;

  /// Leading dimension of C matrix
  int64_t ldc;

  /// Leading dimension of D matrix
  int64_t ldd;

  /// Batch Count
  int batch_count;
};

/// Arguments for (Symm, Hemm)
struct SymmArguments {

  /// Pointer to A matrix
  void const *A;

  /// Pointer to B matrix
  void const *B;

  /// Pointer to C matrix
  void const *C;

  /// Pointer to D matrix
  void *D;

  /// Host or device pointer to alpha scalar
  void const *alpha;

  /// Host or device pointer to beta scalar
  void const *beta;

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode;

  int64_t batch_stride_A;
  int64_t batch_stride_B;
  int64_t batch_stride_C;
  int64_t batch_stride_D;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Two dimensional convolution
//
// OperationKind: Conv2d
//
struct Conv2dConfiguration {

  conv::SplitKMode split_k_mode;
  
  /// Conv2d problem size 
  //  contains strictly conv2d size (N,H,W,C,K,R,S,P,Q,padding,stride,dilation,mode)
  //  also includes (split_k_slices, groups)
  conv::Conv2dProblemSize problem_size;

  // stride of operand A
  std::vector<int64_t> stride_a;

  // stride of operand B
  std::vector<int64_t> stride_b;

  // stride of operand C
  std::vector<int64_t> stride_c;
};


/// Three dimensional convolution
//
// OperationKind: Conv3d
//
struct Conv3dConfiguration {

  conv::SplitKMode split_k_mode;
  
  /// Conv2d problem size 
  //  contains strictly conv2d size (N,D,H,W,C,K,T,R,S,Z,P,Q,padding,stride,dilation,mode)
  //  also includes (split_k_slices, groups)
  conv::Conv3dProblemSize problem_size;

  /// Layout object for activations tensor
  layout::TensorNDHWC layout_activations;

  /// Layout object for filters tensor
  layout::TensorNDHWC layout_filters;

  /// Layout object for source tensor
  layout::TensorNDHWC layout_source;

  /// Layout object for output tensor
  layout::TensorNDHWC layout_output;

  //
  // Methods 
  //

  // Mapping functions (A,B,C -> activation,filter,output)
  layout::TensorNDHWC layout_a(library::ConvKind const &conv_kind) const {
    switch (conv_kind) {
      case library::ConvKind::kFprop: return layout_activations;
      case library::ConvKind::kDgrad: return layout_output;
      case library::ConvKind::kWgrad: return layout_output;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

  layout::TensorNDHWC layout_b(library::ConvKind const &conv_kind) const {
    switch (conv_kind) {
      case library::ConvKind::kFprop: return layout_filters;
      case library::ConvKind::kDgrad: return layout_filters;
      case library::ConvKind::kWgrad: return layout_activations;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }

  layout::TensorNDHWC layout_c(library::ConvKind const &conv_kind) const {
    switch (conv_kind) {
      case library::ConvKind::kFprop: return layout_output;
      case library::ConvKind::kDgrad: return layout_activations;
      case library::ConvKind::kWgrad: return layout_filters;
      default : throw std::runtime_error("Invalid Conv Operator (fprop, dgrad, wgrad)");
    }
  }
};

/// Arguments for CONV
struct ConvArguments {

  /////////////////////////////////////////////////////////
  /// ImplicitGemm matrices A, B, C, D
  /////////////////////////////////////////////////////////
  /// pointer to implicit gemm matrix A
  void const *A;

  /// pointer to implicit gemm matrix B
  void const *B;

  /// pointer to implicit gemm matrix C
  void const *C;

  /// pointer to implicit gemm desitination matrix D
  void *D;

  /// Host or device pointer to alpha scalar
  void const *alpha;

  /// Host or device pointer to beta scalar
  void const *beta;

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for Reduction operations
//
// OperationKind: Reduction
//
struct ReductionConfiguration {

  /// Redcution problem size
  MatrixCoord problem_size;

  /// Number of partitions to reduce
  int partitions;

  /// Number of lements between each partition
  int64_t partition_stride;

  /// leading dimension of 'w'orksace operand
  int64_t ldw; 

  /// leading dimension of 's'ource operand
  int64_t lds;

  /// leading dimension of 'd'estination operand
  int64_t ldd;
};

/// Arguments for Reduction
struct ReductionArguments {

  /// Pointer to workspace matrix
  void const *workspace;

  /// Pointer to source matrix
  void const *source;

  /// Pointer to destination matrix
  void *destination;

  /// pointer to reference matrix
  void *reference;

  /// Host or device pointer to alpha scalar
  void const *alpha;

  /// Host or device pointer to beta scalar
  void const *beta;

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode;
};

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
