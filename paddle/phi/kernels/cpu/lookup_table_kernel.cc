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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

constexpr int64_t kNoPadding = -1;

template <typename T, typename Context>
void LookupTableKernel(const Context &dev_ctx,
                       const DenseTensor &w,
                       const DenseTensor &ids_in,
                       bool is_sparse,
                       bool is_distributed UNUSED,
                       int64_t padding_idx,
                       bool remote_prefetch UNUSED,
                       const std::string &entry_config UNUSED,
                       bool is_test,
                       const std::string &entry UNUSED,
                       const std::string &table_class UNUSED,
                       const std::vector<std::string> &table_names UNUSED,
                       int trainer_id UNUSED,
                       bool grad_inplace UNUSED,
                       const std::vector<std::string> &epmap UNUSED,
                       const std::vector<int64_t> &height_sections UNUSED,
                       DenseTensor *out) {
  auto *ids_t = &ids_in;  // int tensor
  auto *output_t = out;   // float tensor

  int64_t *ids = const_cast<int64_t *>(ids_t->data<int64_t>());
  int64_t ids_numel = ids_t->numel();

  auto *table_t = &w;
  int64_t row_number = table_t->dims()[0];
  int64_t row_width = table_t->dims()[1];

  auto *table = table_t->data<T>();
  auto *output = dev_ctx.template Alloc<T>(output_t);

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
      memcpy(output + i * row_width,
             table + ids[i] * row_width,
             row_width * sizeof(T));
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(lookup_table,
                   CPU,
                   ALL_LAYOUT,
                   phi::LookupTableKernel,
                   float,
                   double,
                   int8_t,
                   int16_t,
                   phi::dtype::bfloat16) {}
