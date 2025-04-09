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

#include "paddle/phi/kernels/embedding_with_scaled_gradient_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/embedding_util.h"

namespace phi {

template <typename T, typename Context>
struct EmbeddingWithScaledGradientGradCPUFunctor {
  EmbeddingWithScaledGradientGradCPUFunctor(const Context& dev_ctx,
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

      std::unordered_map<int64_t, int> count_ids_table;
      std::vector<int64_t> ids_unique;
      for (int64_t i = 0; i < ids_num; ++i) {
        if (count_ids_table.find(ids_data[i]) != count_ids_table.end()) {
          count_ids_table[ids_data[i]]++;
        } else {
          count_ids_table[ids_data[i]] = 1;
          ids_unique.push_back(ids_data[i]);
        }
      }
      auto ids_unique_num = static_cast<int64_t>(ids_unique.size());
      for (int64_t i = 0; i < ids_unique_num; ++i) {
        if (padding_idx_ != kNoPadding && ids_unique[i] == padding_idx_) {
          // do nothing
        } else {
          for (int j = 0; j < D; ++j) {
            d_table_data[ids_unique[i] * D + j] /=
                static_cast<T>(count_ids_table[ids_unique[i]]);
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
void EmbeddingWithScaledGradientGradKernel(const Context& ctx,
                                           const DenseTensor& input,
                                           const DenseTensor& weight,
                                           const DenseTensor& out_grad,
                                           int64_t padding_idx,
                                           DenseTensor* weight_grad) {
  EmbeddingWithScaledGradientGradCPUFunctor<T, Context> functor(
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

PD_REGISTER_KERNEL(embedding_with_scaled_gradient_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::EmbeddingWithScaledGradientGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
