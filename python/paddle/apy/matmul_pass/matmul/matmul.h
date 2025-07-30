// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iostream>
#include <map>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"

#include "cutlass_patch/batched_matrix_coord.h"

#define CHECK_CUTLASS(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

#define CHECK_CUDA(func)                                                      \
  {                                                                           \
    cudaError_t err = func;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << ", " << __FUNCTION__ \
                << "] "                                                       \
                << "CUDA error(" << err << "), " << cudaGetErrorString(err)   \
                << " when call " << #func << std::endl;                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

#define ASSERT_CHECK(__cond)                                            \
  do {                                                                  \
    const bool __cond_var = (__cond);                                   \
    if (!__cond_var) {                                                  \
      ::std::string __err_msg = ::std::string("`") + #__cond +          \
                                "` check failed at " + __FILE__ + ":" + \
                                ::std::to_string(__LINE__);             \
      throw std::runtime_error(__err_msg);                              \
    }                                                                   \
  } while (0)

namespace ap {

template <typename T, int Dim>
struct Alignment {
  static constexpr int kValue =
      ((Dim % 8) == 0) ? 8
                       : (((Dim % 4) == 0) ? 4 : (((Dim % 2) == 0) ? 2 : 1));
};

template <int Dim>
struct Alignment<float, Dim> {
  static constexpr int kValue =
      ((Dim % 4) == 0) ? 4 : (((Dim % 2) == 0) ? 2 : 1);
};

template <typename T, int N>
using Array = cutlass::Array<T, N>;

using MatrixCoord = cutlass::BatchedMatrixCoord;

struct GemmEpilogueParams {
  int batch_count;
  int m;
  int n;
  int k;

  bool transpose_a;
  bool transpose_b;

  // Shape related aruguments
  struct ShapeArguments {
    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;
    int64_t lda;
    int64_t ldb;
    int64_t ldc_bias;
    int64_t ldd;
  };

  ShapeArguments shape_args;

  const void *input;
  const void *weight;
  const void *bias;
  void *output;

  cudaStream_t stream;

  std::vector<int64_t> input0_shape;
  std::vector<int64_t> input1_shape;
  std::vector<const void *> epilogue_in_ptrs;
  std::vector<void *> epilogue_out_ptrs;
  std::vector<std::vector<int64_t>> epilogue_in_shapes;
  std::vector<std::vector<int64_t>> epilogue_out_shapes;

  GemmEpilogueParams() {}
  GemmEpilogueParams(cudaStream_t stream,
                     const void *input,
                     const void *weight,
                     const void *bias,
                     void *output,
                     const std::vector<int64_t> &input_shape,
                     const std::vector<int64_t> &weight_shape,
                     const std::vector<int64_t> &bias_shape,
                     bool transpose_a = false,
                     bool transpose_b = false)
      : stream(stream),
        input(input),
        weight(weight),
        bias(bias),
        output(output),
        transpose_a(transpose_a),
        transpose_b(transpose_b) {
    ASSERT_CHECK(input_shape.size() >= 2U);
    ASSERT_CHECK(weight_shape.size() >= 2U);

    input0_shape = input_shape;
    input1_shape = weight_shape;

    batch_count = 1;
    for (size_t i = 0; i < input_shape.size() - 2; ++i) {
      batch_count *= input_shape[i];
    }

    if (transpose_a) {
      m = input_shape[input_shape.size() - 1];
      k = input_shape[input_shape.size() - 2];
    } else {
      m = input_shape[input_shape.size() - 2];
      k = input_shape[input_shape.size() - 1];
    }
    if (transpose_b) {
      ASSERT_CHECK(weight_shape[weight_shape.size() - 1] == k);
      n = weight_shape[weight_shape.size() - 2];
    } else {
      ASSERT_CHECK(weight_shape[weight_shape.size() - 2] == k);
      n = weight_shape[weight_shape.size() - 1];
    }

    if (bias) {
      ASSERT_CHECK(bias_shape.size() >= 1U);
      ASSERT_CHECK(bias_shape[bias_shape.size() - 1] == n);
    }

#if AP_ENABLE_DEBUG
    std::cout << "-- [GemmEpilogueParams] batch_count: " << batch_count
              << ", m: " << m << ", n: " << n << ", k: " << k << std::endl;
    std::cout << "-- [GemmEpilogueParams] input: " << input << std::endl;
    std::cout << "-- [GemmEpilogueParams] weight: " << weight << std::endl;
    std::cout << "-- [GemmEpilogueParams] bias: " << bias << std::endl;
    std::cout << "-- [GemmEpilogueParams] output: " << output << std::endl;
    std::cout << "-- [GemmEpilogueParams] stream: " << stream << std::endl;
#endif

    shape_args.batch_stride_A = m * k;
    shape_args.batch_stride_B = (weight_shape.size() == 2) ? 0 : n * k;
    shape_args.batch_stride_D = m * n;

    shape_args.lda = transpose_a ? m : k;
    shape_args.ldb = transpose_b ? k : n;
    shape_args.ldd = n;

    bool is_C_bias = bias_shape.size() == 1UL;

    /// Only available in RRR format
    shape_args.batch_stride_C = (!bias || is_C_bias) ? 0 : m * n;
    shape_args.ldc_bias = (!bias || is_C_bias) ? 0 : n;
  }

  void SetEpilogues(const std::vector<const void *> &in_ptrs,
                    const std::vector<void *> &out_ptrs) {
    epilogue_in_ptrs = in_ptrs;
    epilogue_out_ptrs = out_ptrs;
  }

  void SetEpilogueAndShapes(
      const std::vector<const void *> &in_ptrs,
      const std::vector<std::vector<int64_t>> &in_shapes,
      const std::vector<void *> &out_ptrs,
      const std::vector<std::vector<int64_t>> &out_shapes) {
    ASSERT_CHECK(in_ptrs.size() == in_shapes.size());
    epilogue_in_ptrs = in_ptrs;
    epilogue_in_shapes = in_shapes;
    ASSERT_CHECK(out_ptrs.size() == out_shapes.size());
    epilogue_out_ptrs = out_ptrs;
    epilogue_out_shapes = out_shapes;
  }
};

struct GemmBroadcastEpilogueParams : GemmEpilogueParams {
  bool need_broadcast;
  void *broadcast;
  void *broadcast_out;

  GemmBroadcastEpilogueParams(cudaStream_t stream,
                              const void *input,
                              const void *weight,
                              const void *bias,
                              void *broadcast,
                              void *broadcast_out,
                              void *output,
                              const std::vector<int64_t> &input_shape,
                              const std::vector<int64_t> &weight_shape,
                              const std::vector<int64_t> &bias_shape,
                              bool need_broadcast,
                              bool transpose_a = false,
                              bool transpose_b = false)
      : GemmEpilogueParams(stream,
                           input,
                           weight,
                           bias,
                           output,
                           input_shape,
                           weight_shape,
                           bias_shape,
                           transpose_a,
                           transpose_b),
        need_broadcast(need_broadcast),
        broadcast(broadcast),
        broadcast_out(broadcast_out) {}
};

// Convert CUDA data type to cutlass data type
template <typename T>
struct CutlassDataType {
  using Type = T;
};

template <>
struct CutlassDataType<half> {
  using Type = cutlass::half_t;
};

template <>
struct CutlassDataType<__nv_bfloat16> {
  using Type = cutlass::bfloat16_t;
};

// Convert to cutlass layout
template <bool Transposed>
struct MatrixLayout {
  using Type = cutlass::layout::RowMajor;
};

template <>
struct MatrixLayout<true> {
  using Type = cutlass::layout::ColumnMajor;
};

}  // namespace ap
