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

#include "paddle/phi/kernels/embedding_bag_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/embedding_util.h"
namespace phi {

template <typename T, typename Context>
struct EmbeddingBagCPUFunctor {
  EmbeddingBagCPUFunctor(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& weight,
                         const DenseTensor& per_sample_weight,
                         const int64_t padding_idx,
                         const std::string& mode,
                         DenseTensor* out)
      : dev_ctx_(dev_ctx),
        input_(input),
        weight_(weight),
        per_sample_weight_(per_sample_weight),
        padding_idx_(padding_idx),
        mode_(mode),
        out_(out) {}

  using EigenArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
  using EigenVectorMap = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>;
  using ConstEigenVectorMap = Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>;
  using EigenIndex = Eigen::Index;

  template <typename IdT>
  void apply() {
    dev_ctx_.template Alloc<T>(out_);
    const EigenIndex bag_number = input_.dims()[0];
    const EigenIndex sequence_length = input_.dims()[1];
    const EigenIndex output_dim = weight_.dims()[1];

    auto* input_d = input_.data<IdT>();

    auto* weight_d = weight_.data<T>();
    auto* per_sample_weight_d = per_sample_weight_.data<T>();

    auto* output_d = out_->data<T>();

    for (EigenIndex bag = 0; bag < bag_number; ++bag) {
      EigenVectorMap output_slice(&output_d[bag * output_dim], output_dim);
      output_slice.setZero();
      for (EigenIndex seq = 0; seq < sequence_length; ++seq) {
        const ConstEigenVectorMap weight_slice(
            &weight_d[input_d[bag * sequence_length + seq] * output_dim],
            output_dim);
        output_slice += weight_slice * per_sample_weight_d[bag * sequence_length + seq];
      }
      if (mode_ == "mean") {
        output_slice /= static_cast<T>(sequence_length);
      }
    }
  }

 private:
  const Context& dev_ctx_;
  const DenseTensor& input_;
  const DenseTensor& weight_;
  const DenseTensor& per_sample_weight_;
  const int64_t padding_idx_;
  const std::string& mode_;
  DenseTensor* out_;
};
template <typename T, typename Context>
void EmbeddingBagKernel(const Context& ctx,
                        const DenseTensor& input,
                        const DenseTensor& weight,
                        const DenseTensor& per_sample_weight,
                        int64_t padding_idx,
                        const std::string& mode,
                        DenseTensor* out) {
  EmbeddingBagCPUFunctor<T, Context> functor(
      ctx, input, weight, per_sample_weight, padding_idx, mode, out);
  if (input.dtype() == phi::DataType::INT32) {
    functor.template apply<int>();
  } else if (input.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "embebddingbag input only support int32 and int64"));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(embedding_bag,
                   CPU,
                   ALL_LAYOUT,
                   phi::EmbeddingBagKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}
