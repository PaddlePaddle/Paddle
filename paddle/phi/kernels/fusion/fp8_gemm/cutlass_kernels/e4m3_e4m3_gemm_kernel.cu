// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fp8_common.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/float8.h"


namespace phi{
namespace fusion{
namespace cutlass_internal{

template <typename InputType, typename OutType >
bool dispatch_gemm_to_cutlass(GemmEpilogueAllParams params){

using ElementInputA =
    typename std::conditional_t<std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
                                cutlass::float_e4m3_t,
                                cutlass::float_e5m2_t>;
using ElementInputB =
    typename std::conditional_t<std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
                                cutlass::float_e4m3_t,
                                cutlass::float_e5m2_t>;
using ElementOutput =
    typename std::conditional_t<std::is_same_v<OutType, phi::dtype::bfloat16>,
                                cutlass::bfloat16_t,
                                cutlass::half_t>;

using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;
static int const kAlignmentA = 16;
static int const kAlignmentB = 16;


// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm89;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 256, 64>;  // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 64
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;  // <- MMA Op tile M = 16, N = 8, K = 32

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::GemmUniversal<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages,
                                         kAlignmentA,
                                         kAlignmentB,
                                         cutlass::arch::OpMultiplyAdd>;

  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;
  cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
  cutlass::gemm::GemmCoord problem_size = cutlass::gemm::GemmCoord{params.M, params.N, params.K};

  using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(ElementCompute(params.scale), ElementCompute(1.0));
  typename Gemm::Arguments arguments{
      mode,
      problem_size,
      /* batch_count = */ 1,
      epilogue_op,
      reinterpret_cast<ElementInputA*>(const_cast<void*>(params.A)),
      reinterpret_cast<ElementInputB*>(const_cast<void*>(params.B)),
      nullptr,
      reinterpret_cast<ElementOutput*>(params.D),
      params.M* params.K,
      params.K* params.N,
      (int64_t)0,
      params.M* params.N,
      params.lda,
      params.ldb,
      (int64_t)0,
      params.ldd,
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::can_implement() failed" << std::endl;
      return false;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    phi::Allocator* allocator = paddle::GetAllocator(params.place);
    auto workspace = allocator->Allocate(workspace_size);

  std::cout<<"workspace_size: "  << workspace_size <<std::endl;
  std::cout<<"workspace.place(): "  << workspace->place() <<std::endl;
  std::cout<<"workspace.size(): "  << workspace->size() <<std::endl;

    status = gemm_op.initialize(arguments, workspace->ptr(),params.stream);

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::initialize() failed" << std::endl;
      return false;
    }

    //
    // Run the GEMM
    //

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::run() failed" << std::endl;
      return false;
    }

    cudaError_t cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
      return false;
    }
    return true;
}

template <typename InputType, typename OutType >
bool dispatch_gemm_to_cutlass_bias(GemmEpilogueAllParams params){

using ElementInputA =
    typename std::conditional_t<std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
                                cutlass::float_e4m3_t,
                                cutlass::float_e5m2_t>;
using ElementInputB =
    typename std::conditional_t<std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
                                cutlass::float_e4m3_t,
                                cutlass::float_e5m2_t>;
using ElementOutput =
    typename std::conditional_t<std::is_same_v<OutType, phi::dtype::bfloat16>,
                                cutlass::bfloat16_t,
                                cutlass::half_t>;

using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;
static int const kAlignmentA = 16;
static int const kAlignmentB = 16;


// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm89;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 256, 64>;  // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 64
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;  // <- MMA Op tile M = 16, N = 8, K = 32

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::GemmUniversal<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages,
                                         kAlignmentA,
                                         kAlignmentB,
                                         cutlass::arch::OpMultiplyAdd>;

  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;
  cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
  cutlass::gemm::GemmCoord problem_size = cutlass::gemm::GemmCoord{params.M, params.N, params.K};

  using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(ElementCompute(params.scale), ElementCompute(1.0));
  typename Gemm::Arguments arguments{
      mode,
      problem_size,
      /* batch_count = */ 1,
      epilogue_op,
      reinterpret_cast<ElementInputA*>(const_cast<void*>(params.A)),
      reinterpret_cast<ElementInputB*>(const_cast<void*>(params.B)),
      reinterpret_cast<ElementInputB*>(const_cast<void*>(params.bias)),
      reinterpret_cast<ElementOutput*>(params.D),
      params.M* params.K,
      params.K* params.N,
      (int64_t)params.M,
      params.M* params.N,
      params.lda,
      params.ldb,
      (int64_t)0,
      params.ldd,
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::can_implement() failed" << std::endl;
      return false;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    phi::Allocator* allocator = paddle::GetAllocator(params.place);
    auto workspace = allocator->Allocate(workspace_size);

  std::cout<<"workspace_size: "  << workspace_size <<std::endl;
  std::cout<<"workspace.place(): "  << workspace->place() <<std::endl;
  std::cout<<"workspace.size(): "  << workspace->size() <<std::endl;

    status = gemm_op.initialize(arguments, workspace->ptr(),params.stream);

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::initialize() failed" << std::endl;
      return false;
    }

    //
    // Run the GEMM
    //

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::run() failed" << std::endl;
      return false;
    }

    cudaError_t cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
      return false;
    }
    return true;
}

bool FP8FP8GemmScale(GemmEpilogueAllParams params){
    if((params.input_dtype == "e4m3")&&(params.output_dtype == "bf16")){
        dispatch_gemm_to_cutlass<phi::dtype::float8_e4m3fn, phi::dtype::bfloat16>(params);
    }else if((params.input_dtype == "e4m3")&&(params.output_dtype == "fp16")){
        dispatch_gemm_to_cutlass<phi::dtype::float8_e4m3fn, phi::dtype::float16>(params);
    }else if((params.input_dtype == "e5m2")&&(params.output_dtype == "bf16")){
        dispatch_gemm_to_cutlass<phi::dtype::float8_e5m2, phi::dtype::bfloat16>(params);
    }else if((params.input_dtype == "e5m2")&&(params.output_dtype == "fp16")){
        dispatch_gemm_to_cutlass<phi::dtype::float8_e5m2, phi::dtype::float16>(params);
    }
    return false;
}

bool FP8FP8GemmScaleBias(GemmEpilogueAllParams params){
    if((params.input_dtype == "e4m3")&&(params.output_dtype == "bf16")){
        dispatch_gemm_to_cutlass_bias<phi::dtype::float8_e4m3fn, phi::dtype::bfloat16>(params);
    }else if((params.input_dtype == "e4m3")&&(params.output_dtype == "fp16")){
        dispatch_gemm_to_cutlass_bias<phi::dtype::float8_e4m3fn, phi::dtype::float16>(params);
    }else if((params.input_dtype == "e5m2")&&(params.output_dtype == "bf16")){
        dispatch_gemm_to_cutlass_bias<phi::dtype::float8_e5m2, phi::dtype::bfloat16>(params);
    }else if((params.input_dtype == "e5m2")&&(params.output_dtype == "fp16")){
        dispatch_gemm_to_cutlass_bias<phi::dtype::float8_e5m2, phi::dtype::float16>(params);
    }
    return false;
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
