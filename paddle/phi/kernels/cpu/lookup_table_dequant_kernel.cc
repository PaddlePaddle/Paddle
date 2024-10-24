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

#include <string>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"

namespace phi {

template <typename T>
void dequant(const unsigned char *in,
             T *out,
             float min,
             float max,
             int emb_size,
             int pow_2_bits) {
  float scale = (max - min) / pow_2_bits;
  for (int i = 0; i < emb_size; ++i) {
    T x = scale * static_cast<int>(in[i]) + min;
    out[i] = x;
  }
}

constexpr int64_t kNoPadding = -1;

template <typename T, typename Context>
void LookupTableDequantKernel(const Context &dev_ctx,
                              const DenseTensor &w,
                              const DenseTensor &ids_in,
                              int64_t padding_idx,
                              DenseTensor *out) {
  auto *ids_t = &ids_in;  // int tensor
  auto *output_t = out;   // float tensor

  auto *ids = ids_t->data<int64_t>();
  int64_t ids_numel = ids_t->numel();

  auto *table_t = &w;
  int64_t row_number = table_t->dims()[0];
  int64_t quant_number = table_t->dims()[1];
  int64_t row_width = (quant_number - 2) * 4;

  auto *table = table_t->data<float>();
  auto *output = dev_ctx.template Alloc<T>(output_t);
  int pow_2_bits = static_cast<int>(pow(2, 8));
  for (int64_t i = 0; i < ids_numel; ++i) {
    if (padding_idx != kNoPadding && ids[i] == padding_idx) {
      memset(output + i * row_width, 0, row_width * sizeof(T));
    } else {
      PADDLE_ENFORCE_LT(
          ids[i],
          row_number,
          common::errors::InvalidArgument(
              "Variable value (input) of OP(fluid.layers.embedding) "
              "expected >= 0 and < %ld, but got %ld. Please check input "
              "value.",
              row_number,
              ids[i]));
      PADDLE_ENFORCE_GE(
          ids[i],
          0,
          common::errors::InvalidArgument(
              "Variable value (input) of OP(fluid.layers.embedding) "
              "expected >= 0 and < %ld, but got %ld. Please check input "
              "value.",
              row_number,
              ids[i]));
      float min = *(table + ids[i] * quant_number);
      float max = *(table + ids[i] * quant_number + 1);
      int offset = ids[i] * quant_number + 2;
      const unsigned char *tensor_buf =
          reinterpret_cast<const unsigned char *>(table + offset);
      dequant(
          tensor_buf, output + i * row_width, min, max, row_width, pow_2_bits);
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(lookup_table_dequant,
                   CPU,
                   ALL_LAYOUT,
                   phi::LookupTableDequantKernel,
                   float) {}
