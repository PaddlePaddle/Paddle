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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/embedding_util.h"

namespace phi {

template <typename T, typename Context>
struct EmbeddingGradCPUFunctor {
  EmbeddingGradCPUFunctor(const Context& dev_ctx,
                          const DenseTensor& input,
                          const DenseTensor& weight,
                          const DenseTensor& out_grad,
                          int64_t padding_idx,
                          DenseTensor* weight_grad)
      : dev_ctx_(dev_ctx),
        input_(input),
        weight_(weight),
        out_grad_(out_grad),
        weight_grad_(weight_grad),
        padding_idx_(padding_idx) {}

  template <typename IdT>
  void apply() {
    DDim table_dim = weight_.dims();

    auto ids = CopyIdsToVector<IdT, int64_t>(input_);
    auto ids_num = static_cast<int64_t>(ids.size());

    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    {
      auto* d_output = &out_grad_;
      auto* ids_data = ids.data();

      int64_t N = table_dim[0];
      int64_t D = table_dim[1];

      auto* d_output_data = d_output->template data<T>();

      dev_ctx_.template Alloc<T>(weight_grad_);
      auto* d_table_data = weight_grad_->data<T>();

      memset(d_table_data, 0, weight_grad_->numel() * sizeof(T));
      for (int64_t i = 0; i < ids_num; ++i) {
        if (padding_idx_ != kNoPadding && ids_data[i] == padding_idx_) {
          // the gradient of padding_idx should be 0, already done by memset, so
          // do nothing.
        } else {
          PADDLE_ENFORCE_LT(
              ids_data[i],
              N,
              common::errors::InvalidArgument(
                  "Variable value (input) of "
                  "OP(paddle.nn.functional.embedding) "
                  "expected >= 0 and < %ld, but got %ld. Please check input "
                  "value.",
                  N,
                  ids_data[i]));
          PADDLE_ENFORCE_GE(
              ids_data[i],
              0,
              common::errors::InvalidArgument(
                  "Variable value (input) of "
                  "OP(paddle.nn.functional.embedding) "
                  "expected >= 0 and < %ld, but got %ld. Please check input "
                  "value.",
                  N,
                  ids_data[i]));
          for (int j = 0; j < D; ++j) {
            d_table_data[ids_data[i] * D + j] += d_output_data[i * D + j];
          }
        }
      }
    }
  }

 private:
  const Context& dev_ctx_;
  const DenseTensor& input_;
  const DenseTensor& weight_;
  const DenseTensor& out_grad_;
  DenseTensor* weight_grad_;
  int64_t padding_idx_;
};

template <typename T, typename Context>
void EmbeddingGradKernel(const Context& ctx,
                         const DenseTensor& input,
                         const DenseTensor& weight,
                         const DenseTensor& out_grad,
                         int64_t padding_idx,
                         DenseTensor* weight_grad) {
  EmbeddingGradCPUFunctor<T, Context> functor(
      ctx, input, weight, out_grad, padding_idx, weight_grad);
  if (input.dtype() == phi::DataType::INT32) {
    functor.template apply<int>();
  } else if (input.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "embedding input only support int32 and int64"));
  }
}

template <typename T, typename Context>
struct EmbeddingSparseGradCPUFunctor {
  EmbeddingSparseGradCPUFunctor(const Context& dev_ctx,
                                const DenseTensor& input,
                                const DenseTensor& weight,
                                const DenseTensor& out_grad,
                                int64_t padding_idx,
                                SelectedRows* weight_grad)
      : dev_ctx_(dev_ctx),
        input_(input),
        weight_(weight),
        out_grad_(out_grad),
        weight_grad_(weight_grad),
        padding_idx_(padding_idx) {}

  template <typename IdT>
  void apply() {
    DDim table_dim = weight_.dims();

    auto ids = CopyIdsToVector<IdT, int64_t>(input_);
    auto ids_num = static_cast<int64_t>(ids.size());

    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    auto* d_table = weight_grad_;
    auto* d_output = &out_grad_;
    d_table->set_rows(ids);

    auto* d_table_value = d_table->mutable_value();
    d_table_value->Resize({ids_num, table_dim[1]});

    dev_ctx_.template Alloc<T>(d_table_value);

    d_table->set_height(table_dim[0]);

    auto* d_output_data = d_output->template data<T>();
    auto* d_table_data = d_table_value->template data<T>();

    auto d_output_dims = d_output->dims();
    auto d_output_dims_2d =
        flatten_to_2d(d_output_dims, d_output_dims.size() - 1);
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

 private:
  const Context& dev_ctx_;
  const DenseTensor& input_;
  const DenseTensor& weight_;
  const DenseTensor& out_grad_;
  SelectedRows* weight_grad_;
  int64_t padding_idx_;
};

template <typename T, typename Context>
void EmbeddingSparseGradKernel(const Context& ctx,
                               const DenseTensor& input,
                               const DenseTensor& weight,
                               const DenseTensor& out_grad,
                               int64_t padding_idx,
                               SelectedRows* weight_grad) {
  EmbeddingSparseGradCPUFunctor<T, Context> functor(
      ctx, input, weight, out_grad, padding_idx, weight_grad);
  if (input.dtype() == phi::DataType::INT32) {
    functor.template apply<int>();
  } else if (input.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "embedding input only support int32 and int64"));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(embedding_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::EmbeddingGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(embedding_sparse_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::EmbeddingSparseGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
