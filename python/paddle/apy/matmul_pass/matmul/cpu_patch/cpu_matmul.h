#pragma once

#include "cpu_patch/gemm_with_variadic.h"
#include "cpu_patch/epilogue.h"
#include "params.h"
#include "cpu_patch/batched_matrix_coord.h"
#include "cpu_patch/all_tuning_configs.h"

#define __forceinline__ __attribute__((always_inline))
#define __host__
#define __device__

using MatrixCoord = gops::BatchedMatrixCoord;

template <typename T, int VecSize>
using VectorType = gops::VectorType<T, VecSize>;

template <int NUnroll>
using unroll = gops::unroll<NUnroll>;

template <typename T, int VecSize>
auto load_vector(const T* ptr, int64_t offset, bool valid, int64_t size) {
    return _mm256_loadu_ps(ptr + offset);
}

template <typename Vec, std::size_t I>
constexpr const auto& extract_scalar(const Vec& vec, gops::Number<I> i) {
    return vec[i.value];
}

template <typename Vec, std::size_t I>
constexpr auto& extract_scalar(Vec& vec, gops::Number<I> i) {
    return vec[i.value];
}

namespace ap {

template <typename ElementT,
          typename ElementComputeT,
          template <typename T>
          class VariadicFunctor,
          int AlignA,
          int AlignB,
          int ConfigId = DefaultConfig::kConfigId>
void MatmulAddVariadic(
	const GemmEpilogueParams &params,
	const typename VariadicFunctor<ElementComputeT>::Arguments &variadic_args
) {

    constexpr int kElementsPerAccess = 8;
    using Epilogue = gops::epilogue::VariadicEpilogue<VariadicFunctor, kElementsPerAccess, ElementComputeT>;
    using Gemm = gops::GemmWithVariadic<ElementT, ElementComputeT, Epilogue>;
    using Argument = typename Gemm::Argument;

    Epilogue epilogue{variadic_args};
    Gemm gemm;

    auto argument = Argument(
        params.input,
        params.weight,
        params.output,
        params.batch_count,
        params.m,
        params.n,
        params.k,
        params.shape_args.batch_stride_A,
        params.shape_args.batch_stride_B,
        params.shape_args.batch_stride_D,
        epilogue);

    gemm.run(argument);
}

} // namespace ap