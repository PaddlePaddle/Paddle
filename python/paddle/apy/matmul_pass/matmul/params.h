// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

inline int CheckedCastToInt(int64_t value) {
  ASSERT_CHECK(value >= 0);
  ASSERT_CHECK(value <= static_cast<int64_t>(std::numeric_limits<int>::max()));
  return static_cast<int>(value);
}

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

  void *stream_ptr;

  std::vector<int64_t> input0_shape;
  std::vector<int64_t> input1_shape;
  std::vector<const void *> epilogue_in_ptrs;
  std::vector<void *> epilogue_out_ptrs;
  std::vector<std::vector<int64_t>> epilogue_in_shapes;
  std::vector<std::vector<int64_t>> epilogue_out_shapes;

  GemmEpilogueParams() {}
  GemmEpilogueParams(void *stream_ptr,
                     const void *input,
                     const void *weight,
                     const void *bias,
                     void *output,
                     const std::vector<int64_t> &input_shape,
                     const std::vector<int64_t> &weight_shape,
                     const std::vector<int64_t> &bias_shape,
                     bool transpose_a = false,
                     bool transpose_b = false)
      : stream_ptr(stream_ptr),
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

    int64_t batch_count_i64 = 1;
    for (size_t i = 0; i < input_shape.size() - 2; ++i) {
      batch_count_i64 *= input_shape[i];
    }
    batch_count = CheckedCastToInt(batch_count_i64);

    int64_t m_i64;
    int64_t n_i64;
    int64_t k_i64;
    if (transpose_a) {
      m_i64 = input_shape[input_shape.size() - 1];
      k_i64 = input_shape[input_shape.size() - 2];
    } else {
      m_i64 = input_shape[input_shape.size() - 2];
      k_i64 = input_shape[input_shape.size() - 1];
    }
    if (transpose_b) {
      ASSERT_CHECK(weight_shape[weight_shape.size() - 1] == k_i64);
      n_i64 = weight_shape[weight_shape.size() - 2];
    } else {
      ASSERT_CHECK(weight_shape[weight_shape.size() - 2] == k_i64);
      n_i64 = weight_shape[weight_shape.size() - 1];
    }
    m = CheckedCastToInt(m_i64);
    n = CheckedCastToInt(n_i64);
    k = CheckedCastToInt(k_i64);

    if (bias) {
      ASSERT_CHECK(bias_shape.size() >= 1U);
      ASSERT_CHECK(bias_shape[bias_shape.size() - 1] == n_i64);
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

    shape_args.batch_stride_A = m_i64 * k_i64;
    shape_args.batch_stride_B = (weight_shape.size() == 2) ? 0 : n_i64 * k_i64;
    shape_args.batch_stride_D = m_i64 * n_i64;

    shape_args.lda = transpose_a ? m_i64 : k_i64;
    shape_args.ldb = transpose_b ? k_i64 : n_i64;
    shape_args.ldd = n_i64;

    bool is_C_bias = bias_shape.size() == 1UL;

    /// Only available in RRR format
    shape_args.batch_stride_C = (!bias || is_C_bias) ? 0 : m_i64 * n_i64;
    shape_args.ldc_bias = (!bias || is_C_bias) ? 0 : n_i64;
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

}  // namespace ap
