// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/embedding_grad_kernel.h"

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void EmbeddingGradKernel(const Context& ctx,
                         const DenseTensor& input,
                         const DenseTensor& weight,
                         const DenseTensor& out_grad,
                         int64_t padding_idx,
                         DenseTensor* weight_grad) {
  DDim table_dim;
  table_dim = weight.dims();

  auto ids_t = &input;
  auto d_output_t = &out_grad;
  auto d_table_t = weight_grad;

  int64_t ids_numel = ids_t->numel();
  PADDLE_ENFORCE_EQ(
      ids_numel <= std::numeric_limits<int32_t>::max(),
      true,
      phi::errors::OutOfRange(
          "Number of ids greater than int32_t::max , please check "
          "number of ids in LookupTableV2GradXPUKernel."));

  auto& dev_ctx = ctx;
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  const int64_t* ids_data;
  if (ids_t->dtype() == phi::DataType::INT64) {
    ids_data = ids_t->data<int64_t>();
  } else {
    int64_t* ids_tt = RAII_GUARD.alloc_l3_or_gm<int64_t>(ids_t->numel());
    int r = xpu::cast<int32_t, int64_t>(
        ctx.x_context(), ids_t->data<int>(), ids_tt, ids_t->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    ids_data = reinterpret_cast<const int64_t*>(ids_tt);
  }

  const T* d_output_data = d_output_t->data<T>();
  T* d_table_data = dev_ctx.template Alloc<T>(d_table_t);
  int xm = d_table_t->dims()[0];
  int ym = static_cast<int>(ids_numel);
  int n = d_table_t->dims()[1];

  int r = xpu::embedding_grad<T, int64_t>(dev_ctx.x_context(),
                                          d_output_data,
                                          ids_data,
                                          d_table_data,
                                          xm,
                                          n,
                                          ym,
                                          padding_idx);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding_grad");
}

template <typename T, typename Context>
void EmbeddingSparseGradKernel(const Context& ctx,
                               const DenseTensor& input,
                               const DenseTensor& weight,
                               const DenseTensor& out_grad,
                               int64_t padding_idx,
                               SelectedRows* weight_grad) {
  DDim table_dim = weight.dims();

  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  std::vector<int64_t> ids(input.numel());
  if (input.dtype() == phi::DataType::INT64) {
    paddle::memory::Copy(CPUPlace(),
                         ids.data(),
                         input.place(),
                         input.data<int64_t>(),
                         sizeof(int64_t) * input.numel());

  } else if (input.dtype() == phi::DataType::INT32) {
    int64_t* id_t = RAII_GUARD.alloc_l3_or_gm<int64_t>(input.numel());
    int r = xpu::cast<int32_t, int64_t>(
        ctx.x_context(), input.data<int>(), id_t, input.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    paddle::memory::Copy(CPUPlace(),
                         ids.data(),
                         input.place(),
                         id_t,
                         sizeof(int64_t) * input.numel());
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "emebdding input only support int32 and int64"));
  }

  auto ids_num = static_cast<int64_t>(input.numel());
  // Since paddings are not trainable and fixed in forward, the gradient of
  // paddings makes no sense and we don't deal with it in backward.
  auto* d_table = weight_grad;
  auto* d_output = &out_grad;
  d_table->set_rows(ids);

  auto* d_table_value = d_table->mutable_value();
  d_table_value->Resize({ids_num, table_dim[1]});

  ctx.template Alloc<T>(d_table_value);

  d_table->set_height(table_dim[0]);

  auto* d_output_data = d_output->template data<T>();
  auto* d_table_data = d_table_value->template data<T>();

  auto d_output_dims = d_output->dims();
  auto d_output_dims_2d =
      flatten_to_2d(d_output_dims, d_output_dims.size() - 1);
  PADDLE_ENFORCE_EQ(d_table_value->dims(),
                    d_output_dims_2d,
                    phi::errors::InvalidArgument(
                        "ShapeError: The shape of lookup_table@Grad and "
                        "output@Grad should be same. "
                        "But received lookup_table@Grad's shape = [%s], "
                        "output@Grad's shape = [%s].",
                        d_table_value->dims(),
                        d_output_dims_2d));
  int r = xpu::copy<T>(
      ctx.x_context(), d_output_data, d_table_data, d_output->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
}
}  // namespace phi

PD_REGISTER_KERNEL(
    embedding_grad, XPU, ALL_LAYOUT, phi::EmbeddingGradKernel, float) {}
PD_REGISTER_KERNEL(embedding_sparse_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::EmbeddingSparseGradKernel,
                   float) {}
