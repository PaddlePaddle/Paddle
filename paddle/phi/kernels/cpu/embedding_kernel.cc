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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/embedding_util.h"
#include "paddle/phi/kernels/p_norm_kernel.h"

namespace phi {

template <typename T, typename Context>
struct EmbeddingCPUFunctor {
  EmbeddingCPUFunctor(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& weight,
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

    int64_t row_number = weight_.dims()[0];
    int64_t row_width = weight_.dims()[1];

    auto* table = weight_.data<T>();

    dev_ctx_.template Alloc<T>(out_);
    auto* output = out_->data<T>();

    for (int64_t i = 0; i < ids_numel; ++i) {
      if (padding_idx_ == kNoPadding && ids[i] != padding_idx_) {
        PADDLE_ENFORCE_LT(
            ids[i],
            row_number,
            phi::errors::InvalidArgument(
                "Variable value (input) of OP(fluid.layers.embedding) "
                "expected >= 0 and < %ld, but got %ld. Please check input "
                "value.",
                row_number,
                ids[i]));
        PADDLE_ENFORCE_GE(
            ids[i],
            0,
            phi::errors::InvalidArgument(
                "Variable value (input) of OP(fluid.layers.embedding) "
                "expected >= 0 and < %ld, but got %ld. Please check input "
                "value.",
                row_number,
                ids[i]));
      }
    }

#if defined(_OPENMP) && !defined(PADDLE_WITH_CUDA)
#pragma omp parallel for
#endif

    for (int64_t i = 0; i < ids_numel; ++i) {
      if (padding_idx_ != kNoPadding && ids[i] == padding_idx_) {
        memset(output + i * row_width, 0, row_width * sizeof(T));
      } else {
        memcpy(output + i * row_width,
               table + ids[i] * row_width,
               row_width * sizeof(T));
      }
    }
  }

 private:
  const Context& dev_ctx_;
  const DenseTensor& input_;
  const DenseTensor& weight_;
  DenseTensor* out_;
  int64_t padding_idx_;
};

template <typename T, typename Context>
void EmbeddingKernel(const Context& ctx,
                     const DenseTensor& input,
                     const DenseTensor& weight,
                     int64_t padding_idx,
                     DenseTensor* out) {
  EmbeddingCPUFunctor<T, Context> functor(ctx, input, weight, padding_idx, out);

  if (input.dtype() == phi::DataType::INT32) {
    functor.template apply<int>();
  } else if (input.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "emebdding input only support int32 and int64, but get %s",
        input.dtype()));
  }
}

template <typename T, typename Context>
struct EmbeddingRenormCPUFunctor {
  EmbeddingRenormCPUFunctor(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            float max_norm,
                            float norm_type,
                            DenseTensor* out)
      : dev_ctx_(dev_ctx),
        x_(x),
        weight_(weight),
        out_(out),
        max_norm_(max_norm),
        norm_type_(norm_type) {}
  template <typename IdT>
  void apply() {
    auto* out_data = dev_ctx_.template Alloc<T>(out_);
    auto indices = CopyIdsToVector<IdT, int64_t>(x_);

    auto sorted_indices = indices;
    std::sort(sorted_indices.begin(), sorted_indices.end());
    auto num_indices = x_.numel();

    DenseTensor row_norm;
    row_norm.Resize({1});
    dev_ctx_.template Alloc<T>(&row_norm);

    for (int64_t i = 0; i < num_indices; ++i) {
      if (i > 0 && sorted_indices[i] == sorted_indices[i - 1]) {
        continue;
      }
      PADDLE_ENFORCE_GE(x_.data<int64_t>()[i],
                        0,
                        phi::errors::InvalidArgument(
                            "Variable value (input) of OP(embedding_renorm) "
                            "expected >= 0, but got %ld. Please check input "
                            "value.",
                            x_.data<int64_t>()[i]));
      auto row = weight_.Slice(sorted_indices[i], sorted_indices[i] + 1);
      PNormKernel<T, Context>(
          dev_ctx_, row, norm_type_, -1, 0.0, false, true, &row_norm);
      auto row_norm_data = row_norm.data<T>();
      if (static_cast<float>(*row_norm_data) > max_norm_) {
        auto scale = max_norm_ / (static_cast<float>(*row_norm_data) + 1e-7);
        row_norm_data[0] *= scale;
        out_data[sorted_indices[i]] = row_norm_data[0];
      }
    }
  }

 private:
  const Context& dev_ctx_;
  const DenseTensor& x_;
  const DenseTensor& weight_;
  DenseTensor* out_;
  float max_norm_;
  float norm_type_;
};

template <typename T, typename Context>
void EmbeddingRenormKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& weight,
                           float max_norm,
                           float norm_type,
                           DenseTensor* out) {
  EmbeddingRenormCPUFunctor<T, Context> functor(
      ctx, x, weight, max_norm, norm_type, out);
  if (x.dtype() == phi::DataType::INT32) {
    functor.template apply<int>();
  } else if (x.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "emebdding input only support int32 and int64, but get %s", x.dtype()));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(embedding_renorm,
                   CPU,
                   ALL_LAYOUT,
                   phi::EmbeddingRenormKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(embedding,
                   CPU,
                   ALL_LAYOUT,
                   phi::EmbeddingKernel,
                   float,
                   double,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
