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

#include "paddle/phi/kernels/embedding_kernel.h"
#include "paddle/phi/kernels/funcs/embedding_util.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {

template <typename T, typename Context>
struct EmbeddingCPUSparseFunctor {
  EmbeddingCPUSparseFunctor(const Context& dev_ctx,
                            const DenseTensor& input,
                            const SelectedRows& weight,
                            int64_t padding_idx,
                            DenseTensor* out)
      : dev_ctx_(dev_ctx),
        input_(input),
        weight_(weight),
        out_(out),
        padding_idx_(padding_idx) {}

  template <typename IdT>
  void apply() {
    auto ids = CopyIdsToVector<IdT, int64_t>(input_);
    auto ids_numel = static_cast<int64_t>(ids.size());

    const auto& table_t = weight_;
    auto output_t = out_;
    int64_t row_width = table_t.value().dims()[1];
    const auto* table = table_t.value().template data<T>();
    auto* output = dev_ctx_.template Alloc<T>(output_t);
    auto input_data_type =
        paddle::framework::TransToProtoVarType(table_t.value().dtype());

    for (int64_t i = 0; i < ids_numel; ++i) {
      if (padding_idx_ != kNoPadding && ids[i] == padding_idx_) {
        memset(output + i * row_width, 0, row_width * sizeof(T));
      } else {
        PADDLE_ENFORCE_GE(
            ids[i],
            0,
            phi::errors::InvalidArgument(
                "Variable value (input) of OP(fluid.layers.embedding) "
                "expected >= 0. But received %ld",
                ids[i]));
        auto id_index = table_t.Index(ids[i]);
        PADDLE_ENFORCE_GE(
            id_index,
            0,
            phi::errors::InvalidArgument(
                "the input key should be exists. But received %d.", id_index));

        if (input_data_type == paddle::framework::proto::VarType::BF16) {
          memcpy(output + i * row_width,
                 table + id_index * row_width,
                 row_width * sizeof(T));
        } else {
          auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(dev_ctx_);
          blas.VCOPY(
              row_width, table + id_index * row_width, output + i * row_width);
        }
      }
    }
  }

 private:
  const Context& dev_ctx_;
  const DenseTensor& input_;
  const SelectedRows& weight_;
  DenseTensor* out_;
  int64_t padding_idx_;
};

template <typename T, typename Context>
void SparseWeightEmbeddingKernel(const Context& ctx,
                                 const DenseTensor& input,
                                 const SelectedRows& weight,
                                 int64_t padding_idx,
                                 DenseTensor* out) {
  EmbeddingCPUSparseFunctor<T, Context> functor(
      ctx, input, weight, padding_idx, out);

  if (input.dtype() == phi::DataType::INT32) {
    functor.template apply<int>();
  } else if (input.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "emebdding input only support int32 and int64"));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sparse_weight_embedding,
                   CPU,
                   ALL_LAYOUT,
                   phi::SparseWeightEmbeddingKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}
