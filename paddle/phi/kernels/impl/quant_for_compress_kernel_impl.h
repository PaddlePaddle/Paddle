// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef PADDLE_PHI_KERNELS_IMPL_QUANT_FOR_COMPRESS_KERNEL_IMPL_H_
#define PADDLE_PHI_KERNELS_IMPL_QUANT_FOR_COMPRESS_KERNEL_IMPL_H_
#include <iostream>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T>
inline T xabs(const T x) {
  return x < static_cast<T>(0.0) ? -x : x;
}

template <typename T>
void per_channel_scale(float* scale, const T* input, size_t m, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    T max = input[i];
    for (size_t j = 0; j < m; ++j) {
      max = xabs(input[j * n + i]) > max ? xabs(input[j * n + i]) : max;
    }
    scale[i] = static_cast<float>(max) / 127.0;
  }
}

template <typename T, typename D>
void per_channel_quant(
    D* output, const T* input, const float* scale, size_t m, size_t n) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      output[i * n + j] = static_cast<D>(
          round(static_cast<float>(input[i * n + j]) / scale[j]));
    }
  }
}

void row_major_to_column_major(int8_t* col_major_tensor,
                               const int8_t* row_major_tensor,
                               const std::vector<size_t>& shape) {
  size_t m = shape[0];
  size_t n = shape[1];
  for (size_t i = 0; i < m * n; i++) {
    size_t im = i / n;
    size_t in = i % n;
    col_major_tensor[in * m + im] = row_major_tensor[im * n + in];
  }
}

void add_bias_and_interleave_int8s_inplace(int8_t* int8_tensor_ptr,
                                           size_t num_elts) {
  int8_t* int8_tensor = reinterpret_cast<int8_t*>(int8_tensor_ptr);
  for (size_t ii = 0; ii < num_elts; ++ii) {
    int8_tensor[ii] =
        static_cast<int8_t>(static_cast<int>(int8_tensor[ii]) + 128);
  }
  // Step 2 will transform the layout of a 32-bit register in CUDA in order to
  // match the int4 layout. This has no performance benefit and is purely so
  // that int4 and int8 have the same layout. Pictorially, this does the
  // following: bit 32                                                      0
  //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 8 bits)
  //
  // And it will rearrange the output 32 bit register to be the following:
  // bit 32                                                      0
  //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)

  for (size_t base = 0; base < num_elts; base += 4) {
    std::swap(int8_tensor[base + 1], int8_tensor[base + 2]);
  }
}

void permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor,
                                   const int8_t* quantized_tensor,
                                   const std::vector<size_t>& shape,
                                   const int64_t arch_version) {
  // We only want to run this step for weight only quant.
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const int BITS_PER_ELT = 8;
  const int K = 16 / BITS_PER_ELT;
  // const int ELTS_PER_BYTE = 8 / BITS_PER_ELT;
  const int ELTS_PER_REG = 32 / BITS_PER_ELT;

  const uint32_t* input_byte_ptr =
      reinterpret_cast<const uint32_t*>(quantized_tensor);
  uint32_t* output_byte_ptr =
      reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

  // int       MMA_SHAPE_N    = 8;
  int B_ROWS_PER_MMA = 8 * K;
  const int elts_in_int32 = 32 / BITS_PER_ELT;

  const int num_vec_cols = num_cols / elts_in_int32;

  // The code is written as below so it works for both int8 and packed int4.
  for (size_t base_row = 0; base_row < num_rows; base_row += B_ROWS_PER_MMA) {
    for (int tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row) {
      for (int write_col = 0; write_col < num_vec_cols; ++write_col) {
        const int write_row = base_row + tile_row;
        const int tile_read_row = 8 * (((tile_row % ELTS_PER_REG) / 2)) +
                                  tile_row % 2 + 2 * (tile_row / ELTS_PER_REG);
        const int read_row = base_row + tile_read_row;
        const int read_col = write_col;

        const int64_t read_offset = int64_t(read_row) * num_vec_cols + read_col;
        const int64_t write_offset =
            int64_t(write_row) * num_vec_cols + write_col;
        output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
      }
    }
  }
}

void interleave_column_major_tensor(int8_t* interleaved_quantized_tensor,
                                    const int8_t* quantized_tensor,
                                    const std::vector<size_t>& shape) {
  // We only want to run this step for weight only quant.
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const size_t BITS_PER_ELT = 8;
  const size_t elts_in_int32 = 32 / BITS_PER_ELT;

  const size_t rows_per_tile = 64;

  const uint32_t* input_byte_ptr =
      reinterpret_cast<const uint32_t*>(quantized_tensor);
  uint32_t* output_byte_ptr =
      reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);

  const size_t num_vec_rows = num_rows / elts_in_int32;
  const size_t vec_rows_per_tile = rows_per_tile / elts_in_int32;
  const size_t interleave = 2;
  for (size_t read_col = 0; read_col < num_cols; ++read_col) {
    const size_t write_col = read_col / interleave;
    for (size_t base_vec_row = 0; base_vec_row < num_vec_rows;
         base_vec_row += vec_rows_per_tile) {
      for (size_t vec_read_row = base_vec_row;
           vec_read_row <
           std::min(num_vec_rows, base_vec_row + vec_rows_per_tile);
           ++vec_read_row) {
        const size_t vec_write_row =
            interleave * base_vec_row +
            vec_rows_per_tile * (read_col % interleave) +
            vec_read_row % vec_rows_per_tile;

        const size_t read_offset =
            size_t(read_col) * num_vec_rows + vec_read_row;
        const size_t write_offset =
            size_t(write_col) * num_vec_rows * interleave + vec_write_row;
        output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
      }
    }
  }
}

}  // namespace phi
#endif  // PADDLE_PHI_KERNELS_IMPL_QUANT_FOR_COMPRESS_KERNEL_IMPL_H_
