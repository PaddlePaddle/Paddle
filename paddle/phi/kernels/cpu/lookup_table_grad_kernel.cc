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
void LookupTableGradKernel(const Context &dev_ctx,
                           const DenseTensor &w,
                           const DenseTensor &ids_in,
                           const DenseTensor &out_grad,
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
                           DenseTensor *w_grad) {
  DDim table_dim;
  table_dim = w.dims();

  // Since paddings are not trainable and fixed in forward, the gradient of
  // paddings makes no sense and we don't deal with it in backward.
  auto *ids = &ids_in;
  auto *d_output = &out_grad;
  auto *d_table = w_grad;

  auto *ids_data = ids->data<int64_t>();

  int64_t N = table_dim[0];
  int64_t D = table_dim[1];

  auto *d_output_data = d_output->data<T>();
  auto *d_table_data = dev_ctx.template Alloc<T>(d_table);

  memset(d_table_data, 0, d_table->numel() * sizeof(T));

  for (int64_t i = 0; i < ids->numel(); ++i) {
    if (padding_idx != kNoPadding && ids_data[i] == padding_idx) {
      // the gradient of padding_idx should be 0, already done by memset, so
      // do nothing.
    } else {
      PADDLE_ENFORCE_LT(
          ids_data[i],
          N,
          common::errors::InvalidArgument(
              "Variable value (input) of OP(fluid.layers.embedding) "
              "expected >= 0 and < %ld, but got %ld. Please check input "
              "value.",
              N,
              ids_data[i]));
      PADDLE_ENFORCE_GE(
          ids_data[i],
          0,
          common::errors::InvalidArgument(
              "Variable value (input) of OP(fluid.layers.embedding) "
              "expected >= 0 and < %ld, but got %ld. Please check input"
              "value.",
              N,
              ids_data[i]));
      for (int j = 0; j < D; ++j) {
        d_table_data[ids_data[i] * D + j] += d_output_data[i * D + j];
      }
    }
  }
}

template <typename T, typename Context>
void LookupTableSparseGradKernel(
    const Context &dev_ctx,
    const DenseTensor &w,
    const DenseTensor &ids_in,
    const DenseTensor &out_grad,
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
    SelectedRows *w_grad) {
  DDim table_dim;
  table_dim = w.dims();

  // Since paddings are not trainable and fixed in forward, the gradient of
  // paddings makes no sense and we don't deal with it in backward.
  auto *ids = &ids_in;
  auto *d_output = &out_grad;
  auto *d_table = w_grad;

  auto *ids_data = ids->data<int64_t>();
  int64_t ids_num = ids->numel();

  std::vector<int64_t> new_rows;
  new_rows.resize(ids_num);
  std::memcpy(&new_rows[0], ids_data, ids_num * sizeof(int64_t));
  d_table->set_rows(new_rows);

  auto *d_table_value = d_table->mutable_value();
  d_table_value->Resize({ids_num, table_dim[1]});
  dev_ctx.template Alloc<T>(d_table_value);
  d_table->set_height(table_dim[0]);

  auto *d_output_data = d_output->data<T>();
  auto *d_table_data = d_table_value->data<T>();

  auto d_output_dims = d_output->dims();
  auto d_output_dims_2d =
      common::flatten_to_2d(d_output_dims, d_output_dims.size() - 1);
  PADDLE_ENFORCE_EQ(d_table_value->dims(),
                    d_output_dims_2d,
                    common::errors::InvalidArgument(
                        "ShapeError: The shape of lookup_table@Grad and "
                        "output@Grad should be same. "
                        "But received lookup_table@Grad's shape = [%s], "
                        "output@Grad's shape = [%s].",
                        d_table_value->dims(),
                        d_output_dims_2d));
  memcpy(d_table_data, d_output_data, sizeof(T) * d_output->numel());
}
}  // namespace phi

PD_REGISTER_KERNEL(lookup_table_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::LookupTableGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(lookup_table_sparse_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::LookupTableSparseGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}
