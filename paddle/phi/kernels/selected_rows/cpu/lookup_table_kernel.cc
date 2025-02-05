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
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace sr {
using DDim = phi::DDim;

constexpr int64_t kNoPadding = -1;

template <typename T, typename Context>
void LookupTableKernel(const Context &dev_ctx,
                       const SelectedRows &w,
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
                       SelectedRows *out) {
  auto *ids_t = &ids_in;  // int tensor
  auto *output_t = out;   // float tensor

  int64_t *ids = const_cast<int64_t *>(ids_t->data<int64_t>());
  int64_t ids_numel = ids_t->numel();

  const auto &table_t = w;
  int64_t row_width = table_t.value().dims()[1];
  const auto *table = table_t.value().data<T>();
  auto *output = dev_ctx.template Alloc<T>(output_t);
  auto input_data_type = table_t.value().dtype();
  for (int64_t i = 0; i < ids_numel; ++i) {
    if (padding_idx != kNoPadding && ids[i] == padding_idx) {
      memset(output + i * row_width, 0, row_width * sizeof(T));
    } else {
      PADDLE_ENFORCE_GE(
          ids[i],
          0,
          common::errors::InvalidArgument(
              "Variable value (input) of OP(fluid.layers.embedding) "
              "expected >= 0. But received %ld",
              ids[i]));
      if (is_test) {
        auto id_index = table_t.GetIndexFromId(ids[i]);

        if (id_index != -1) {
          if (input_data_type == phi::DataType::INT8 ||
              input_data_type == phi::DataType::INT16 ||
              input_data_type == phi::DataType::BFLOAT16) {
            memcpy(output + i * row_width,
                   table + id_index * row_width,
                   row_width * sizeof(T));
          } else {
            auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(dev_ctx);
            blas.VCOPY(row_width,
                       table + id_index * row_width,
                       output + i * row_width);
          }
        } else {
          memset(output + i * row_width, 0, row_width * sizeof(T));
        }
      } else {
        auto id_index = table_t.Index(ids[i]);
        PADDLE_ENFORCE_GE(
            ids[i],
            0,
            common::errors::InvalidArgument(
                "Variable value (input) of OP(fluid.layers.embedding) "
                "expected >= 0. But received %ld",
                ids[i]));
        PADDLE_ENFORCE_GE(
            id_index,
            0,
            common::errors::InvalidArgument(
                "the input key should be exists. But received %d.", id_index));

        if (input_data_type == phi::DataType::INT8 ||
            input_data_type == phi::DataType::INT16 ||
            input_data_type == phi::DataType::BFLOAT16) {
          memcpy(output + i * row_width,
                 table + id_index * row_width,
                 row_width * sizeof(T));
        } else {
          auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(dev_ctx);
          blas.VCOPY(
              row_width, table + id_index * row_width, output + i * row_width);
        }
      }
    }
  }
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(lookup_table_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::LookupTableKernel,
                   float,
                   double,
                   int8_t,
                   int16_t,
                   phi::dtype::bfloat16) {}
