/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

#pragma once

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

template <typename T, typename ScaleT>
void per_channel_scale(
    ScaleT* scale, const T* input, size_t m, size_t n, float bound) {
  for (size_t i = 0; i < n; ++i) {
    float max = static_cast<float>(input[i]);
    for (size_t j = 0; j < m; ++j) {
      max = static_cast<float>(xabs(input[j * n + i])) > max
                ? static_cast<float>(xabs(input[j * n + i]))
                : max;
    }
    scale[i] = static_cast<ScaleT>(max / bound);
  }
}

template <typename T, typename ScaleT>
void group_wise_scale(ScaleT* scale,
                      const T* input,
                      size_t m,
                      size_t n,
                      float bound,
                      size_t group_size) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; j += group_size) {
      float max = static_cast<float>(0.f);
      for (size_t k = 0; k < group_size && j + k < m; ++k) {
        max = static_cast<float>(xabs(input[(j + k) * n + i])) > max
                  ? static_cast<float>(xabs(input[(j + k) * n + i]))
                  : max;
      }
      scale[static_cast<int>(j / group_size) * n + i] =
          static_cast<ScaleT>(max / bound);
    }
  }
}

template <typename T, int quant_bit = 8, typename ScaleT>
void per_channel_quant(int8_t* output,
                       const T* input,
                       const ScaleT* scale,
                       size_t num_rows,
                       size_t num_cols) {
  size_t bytes_per_out_col = num_cols * quant_bit / 8;
  for (size_t ii = 0; ii < num_rows; ++ii) {
    int8_t* current_quantized_weight_row = output + ii * bytes_per_out_col;
    const T* current_weight_row = input + ii * num_cols;
    for (size_t jj = 0; jj < bytes_per_out_col; ++jj) {
      if (quant_bit == 8) {
        const float col_scale = static_cast<float>(scale[jj]);
        const float weight_elt = static_cast<float>(current_weight_row[jj]);
        const float scaled_weight = round(weight_elt / col_scale);
        const int8_t clipped_weight = static_cast<int8_t>(
            std::max(-127.f, std::min(127.f, scaled_weight)));
        current_quantized_weight_row[jj] = clipped_weight;
      } else if (quant_bit == 4) {
        // We will pack two int4 elements per iteration of the inner loop.
        int8_t packed_int4s = 0;
        for (int packed_idx = 0; packed_idx < 2; ++packed_idx) {
          const size_t input_idx = 2 * jj + packed_idx;
          if (input_idx < num_cols) {
            const float col_scale = static_cast<float>(scale[input_idx]);
            const float weight_elt =
                static_cast<float>(current_weight_row[input_idx]);
            const float scaled_weight = round(weight_elt / col_scale);
            int int_weight = static_cast<int>(scaled_weight);
            const int8_t clipped_weight = std::max(-7, std::min(7, int_weight));

            // Kill the sign extension bits (hence 0x0F mask) then shift to
            // upper bits if packing the second int4 and or the bits into the
            // final result.
            packed_int4s |= ((clipped_weight & 0x0F) << (4 * packed_idx));
          }
        }
        current_quantized_weight_row[jj] = packed_int4s;
      } else {
        phi::errors::Unimplemented("Unsupported quantization bits: %d",
                                   quant_bit);
      }
    }
  }
}

template <typename T, int quant_bit = 8, typename ScaleT>
void group_wise_quant(int8_t* output,
                      const T* input,
                      const ScaleT* scale,
                      size_t num_rows,
                      size_t num_cols,
                      const int group_size) {
  size_t bytes_per_out_col = num_cols * quant_bit / 8;
  for (size_t ii = 0; ii < num_rows; ++ii) {
    int8_t* current_quantized_weight_row = output + ii * bytes_per_out_col;
    const T* current_weight_row = input + ii * num_cols;
    for (size_t jj = 0; jj < bytes_per_out_col; ++jj) {
      if (quant_bit == 8) {
        size_t scale_cur_offset = jj + (ii / group_size) * num_cols;
        const float col_scale = static_cast<float>(scale[scale_cur_offset]);
        const float weight_elt = static_cast<float>(current_weight_row[jj]);
        const float scaled_weight = round(weight_elt / col_scale);
        const int8_t clipped_weight = static_cast<int8_t>(
            std::max(-127.f, std::min(127.f, scaled_weight)));
        current_quantized_weight_row[jj] = clipped_weight;
      } else if (quant_bit == 4) {
        // We will pack two int4 elements per iteration of the inner loop.
        int8_t packed_int4s = 0;
        for (int packed_idx = 0; packed_idx < 2; ++packed_idx) {
          const size_t input_idx = 2 * jj + packed_idx;
          if (input_idx < num_cols) {
            size_t scale_cur_offset = input_idx + (ii / group_size) * num_cols;
            const float col_scale = static_cast<float>(scale[scale_cur_offset]);
            const float weight_elt =
                static_cast<float>(current_weight_row[input_idx]);
            const float scaled_weight = round(weight_elt / col_scale);
            int int_weight = static_cast<int>(scaled_weight);
            const int8_t clipped_weight = std::max(-7, std::min(7, int_weight));

            // Kill the sign extension bits (hence 0x0F mask) then shift to
            // upper bits if packing the second int4 and or the bits into the
            // final result.
            packed_int4s |= ((clipped_weight & 0x0F) << (4 * packed_idx));
          }
        }
        current_quantized_weight_row[jj] = packed_int4s;
      } else {
        phi::errors::Unimplemented("Unsupported quantization bits: %d",
                                   quant_bit);
      }
    }
  }
}

template <int quant_bit = 8>
void add_bias_and_interleave_inplace(int8_t* tensor_ptr, size_t num_elts) {
  const size_t num_bytes = num_elts * quant_bit / 8;

  for (size_t ii = 0; ii < num_bytes; ++ii) {
    if (quant_bit == 8) {
      tensor_ptr[ii] =
          static_cast<int8_t>(static_cast<int>(tensor_ptr[ii]) + 128);
    } else {
      int8_t transformed_packed_int4s = 0;
      int8_t transformed_first_elt =
          (int8_t(tensor_ptr[ii] << 4) >> 4) +
          8;  // The double shift here is to ensure sign extension
      int8_t transformed_second_elt = (tensor_ptr[ii] >> 4) + 8;

      if (!(transformed_first_elt >= 0 && transformed_first_elt <= 15)) {
        phi::errors::InvalidArgument(
            "Illegal result for int4 transform (first elt)");
      }
      if (!(transformed_second_elt >= 0 && transformed_second_elt <= 15)) {
        phi::errors::InvalidArgument(
            "Illegal result for int4 transform (second elt)");
      }
      // We don't need to mask in these ops since everything should be in the
      // range 0-15
      transformed_packed_int4s |= transformed_first_elt;
      transformed_packed_int4s |= (transformed_second_elt << 4);
      tensor_ptr[ii] = transformed_packed_int4s;
    }
  }
  if (quant_bit == 8) {
    for (size_t base = 0; base < num_elts; base += 4) {
      std::swap(tensor_ptr[base + 1], tensor_ptr[base + 2]);
    }
  } else {
    const size_t num_registers = num_bytes / 4;

    uint32_t* register_ptr = reinterpret_cast<uint32_t*>(tensor_ptr);
    for (size_t ii = 0; ii < num_registers; ++ii) {
      const uint32_t current_register = register_ptr[ii];
      uint32_t transformed_register = 0;

      for (int dest_idx = 0; dest_idx < 8; ++dest_idx) {
        const int src_idx =
            dest_idx < 4 ? 2 * dest_idx : 2 * (dest_idx - 4) + 1;
        const int src_shift = 4 * src_idx;
        const int dest_shift = 4 * dest_idx;

        const uint32_t src_bits = (current_register >> src_shift) & 0xF;
        transformed_register |= (src_bits << dest_shift);
      }
      register_ptr[ii] = transformed_register;
    }
  }
}

template <int quant_bit>
void permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor,
                                   const int8_t* quantized_tensor,
                                   const std::vector<size_t>& shape) {
  // We only want to run this step for weight only quant.
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const int BITS_PER_ELT = quant_bit;
  const int K = 16 / BITS_PER_ELT;
  const int ELTS_PER_REG = 32 / BITS_PER_ELT;

  const uint32_t* input_byte_ptr =
      reinterpret_cast<const uint32_t*>(quantized_tensor);
  uint32_t* output_byte_ptr =
      reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

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

        const int64_t read_offset =
            static_cast<int64_t>(read_row) * num_vec_cols + read_col;
        const int64_t write_offset =
            static_cast<int64_t>(write_row) * num_vec_cols + write_col;
        output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
      }
    }
  }
}

template <int quant_bit>
void subbyte_transpose_impl(int8_t* transposed_quantized_tensor,
                            const int8_t* quantized_tensor,
                            const std::vector<size_t>& shape) {
  const int bits_per_elt = quant_bit;

  // FT_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be
  // 2-D or 3-D");
  // const size_t num_experts = 1;
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const size_t col_bytes = num_cols * bits_per_elt / 8;
  const size_t col_bytes_trans = num_rows * bits_per_elt / 8;
  // const size_t num_bytes = size_t(num_experts) * num_rows * col_bytes;

  const uint8_t* input_byte_ptr =
      reinterpret_cast<const uint8_t*>(quantized_tensor);
  uint8_t* output_byte_ptr =
      reinterpret_cast<uint8_t*>(transposed_quantized_tensor);

  static constexpr int ELTS_PER_BYTE = 8 / quant_bit;

  static constexpr int M_TILE_L1 = 64;
  static constexpr int N_TILE_L1 = M_TILE_L1 / ELTS_PER_BYTE;
  uint8_t cache_buf[M_TILE_L1][N_TILE_L1];

  static constexpr int VECTOR_WIDTH = std::min(32, N_TILE_L1);

  // const int num_m_tiles = (num_rows + M_TILE_L1 - 1) / M_TILE_L1;
  // const int num_n_tiles = (col_bytes + N_TILE_L1 - 1) / N_TILE_L1;

  for (size_t row_tile_start = 0; row_tile_start < num_rows;
       row_tile_start += M_TILE_L1) {
    for (size_t col_tile_start_byte = 0; col_tile_start_byte < col_bytes;
         col_tile_start_byte += N_TILE_L1) {
      const int row_limit = std::min(row_tile_start + M_TILE_L1, num_rows);
      const int col_limit =
          std::min(col_tile_start_byte + N_TILE_L1, col_bytes);

      for (int ii = 0; ii < M_TILE_L1; ++ii) {
        const int row = row_tile_start + ii;

        for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
          const int col = col_tile_start_byte + jj;

          const size_t logical_src_offset = row * col_bytes + col;

          if (row < row_limit && col < col_limit) {
            for (int v = 0; v < VECTOR_WIDTH; ++v) {
              cache_buf[ii][jj + v] = input_byte_ptr[logical_src_offset + v];
            }
          }
        }
      }

      if (quant_bit == 8) {
        for (int ii = 0; ii < M_TILE_L1; ++ii) {
          for (int jj = ii + 1; jj < N_TILE_L1; ++jj) {
            std::swap(cache_buf[ii][jj], cache_buf[jj][ii]);
          }
        }
      } else if (quant_bit == 4) {
        for (int ii = 0; ii < M_TILE_L1; ++ii) {
          // Using M_TILE_L1 here is deliberate since we assume that the cache
          // tile is square in the number of elements (not necessarily the
          // number of bytes).
          for (int jj = ii + 1; jj < M_TILE_L1; ++jj) {
            const int ii_byte = ii / ELTS_PER_BYTE;
            const int ii_bit_offset = ii % ELTS_PER_BYTE;

            const int jj_byte = jj / ELTS_PER_BYTE;
            const int jj_bit_offset = jj % ELTS_PER_BYTE;

            uint8_t src_elt =
                0xF & (cache_buf[ii][jj_byte] >> (4 * jj_bit_offset));
            uint8_t tgt_elt =
                0xF & (cache_buf[jj][ii_byte] >> (4 * ii_bit_offset));

            cache_buf[ii][jj_byte] &= (0xF0 >> (4 * jj_bit_offset));
            cache_buf[jj][ii_byte] &= (0xF0 >> (4 * ii_bit_offset));

            cache_buf[ii][jj_byte] |= (tgt_elt << (4 * jj_bit_offset));
            cache_buf[jj][ii_byte] |= (src_elt << (4 * ii_bit_offset));
          }
        }
      } else {
        phi::errors::Unimplemented("Unsupported quantization bits: %d",
                                   quant_bit);
      }

      const size_t row_tile_start_trans = col_tile_start_byte * ELTS_PER_BYTE;
      const size_t col_tile_start_byte_trans = row_tile_start / ELTS_PER_BYTE;

      const int row_limit_trans =
          std::min(row_tile_start_trans + M_TILE_L1, num_cols);
      const int col_limit_trans =
          std::min(col_tile_start_byte_trans + N_TILE_L1, col_bytes_trans);

      for (int ii = 0; ii < M_TILE_L1; ++ii) {
        const int row = row_tile_start_trans + ii;
        for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
          const int col = col_tile_start_byte_trans + jj;

          const size_t logical_tgt_offset = row * col_bytes_trans + col;

          if (row < row_limit_trans && col < col_limit_trans) {
            for (int v = 0; v < VECTOR_WIDTH; ++v) {
              output_byte_ptr[logical_tgt_offset + v] = cache_buf[ii][jj + v];
            }
          }
        }
      }
    }
  }
}

template <int quant_bit>
void interleave_column_major_tensor(int8_t* interleaved_quantized_tensor,
                                    const int8_t* quantized_tensor,
                                    const std::vector<size_t>& shape) {
  // We only want to run this step for weight only quant.
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const size_t BITS_PER_ELT = quant_bit;
  const size_t elts_in_int32 = 32 / BITS_PER_ELT;

  const size_t rows_per_tile = 64;

  const uint32_t* input_byte_ptr =
      reinterpret_cast<const uint32_t*>(quantized_tensor);
  uint32_t* output_byte_ptr =
      reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);

  const size_t num_vec_rows = num_rows / elts_in_int32;
  const size_t vec_rows_per_tile = rows_per_tile / elts_in_int32;
  const size_t interleave = 128 * 8 / quant_bit / rows_per_tile;
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
