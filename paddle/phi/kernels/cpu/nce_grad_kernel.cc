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

#include <math.h>

#include <iterator>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math/sampler.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace phi {

using Sampler = phi::math::Sampler;

template <typename T, typename Context>
void NCEGradKernel(const Context &dev_ctx,
                   const DenseTensor &input_in,
                   const DenseTensor &label_in,
                   const paddle::optional<DenseTensor> &bias_in,
                   const DenseTensor &weight_in,
                   const DenseTensor &sample_logits_in,
                   const DenseTensor &sample_labels_in,
                   const paddle::optional<DenseTensor> &sample_weight_in,
                   const paddle::optional<DenseTensor> &custom_dist_probs,
                   const paddle::optional<DenseTensor> &custom_dist_alias,
                   const paddle::optional<DenseTensor> &custom_dist_alias_probs,
                   const DenseTensor &cost_grad,
                   int num_total_classes,
                   const std::vector<int> &custom_neg_classes,
                   int num_neg_samples,
                   int sampler_in,
                   int seed,
                   bool is_sparse,
                   bool remote_prefetch,
                   bool is_test,
                   DenseTensor *input_grad,
                   DenseTensor *bias_grad,
                   DenseTensor *weight_grad) {
  auto d_out = &cost_grad;
  const T *d_out_data = d_out->data<T>();
  auto label = &label_in;
  auto sample_out = &sample_logits_in;
  const T *sample_out_data = sample_out->data<T>();
  auto sample_labels = &sample_labels_in;
  const int64_t *sample_labels_data = sample_labels->data<int64_t>();
  auto sample_weight = sample_weight_in.get_ptr();
  const T *sample_weight_data = nullptr;
  if (sample_weight != nullptr) {
    sample_weight_data = sample_weight->data<T>();
  }
  int num_true_class = 1;
  if (label != nullptr) {
    num_true_class = label->dims()[1];
  }

  int sampler_type = sampler_in;
  Sampler *sampler;
  switch (sampler_type) {
    case 0: {
      sampler = new phi::math::UniformSampler(num_total_classes - 1, seed);
      break;
    }
    case 1: {
      sampler = new phi::math::LogUniformSampler(num_total_classes - 1, seed);
      break;
    }
    case 2: {
      auto dist_probs = custom_dist_probs.get_ptr();
      auto dist_alias = custom_dist_alias.get_ptr();
      auto dist_alias_probs = custom_dist_alias_probs.get_ptr();

      PADDLE_ENFORCE_EQ(
          dist_probs->numel(),
          num_total_classes,
          common::errors::InvalidArgument(
              "ShapeError: The number of elements in Input(CustomDistProbs) "
              "should be equal to the number of total classes. But Received: "
              "Input(CustomDistProbs).numel() = %d, Attr(num_total_classes) "
              "= %d.",
              dist_probs->numel(),
              num_total_classes));
      PADDLE_ENFORCE_EQ(
          dist_alias->numel(),
          num_total_classes,
          common::errors::InvalidArgument(
              "ShapeError: The number of elements in Input(CustomDistAlias) "
              "should be equal to the number of total classes. But Received: "
              "Input(CustomDistAlias).numel() = %d, Attr(num_total_classes) "
              "= %d.",
              dist_alias->numel(),
              num_total_classes));
      PADDLE_ENFORCE_EQ(
          dist_alias_probs->numel(),
          num_total_classes,
          common::errors::InvalidArgument(
              "ShapeError: The number of elements in "
              "Input(CustomDistAliasProbs) "
              "should be equal to the number of total classes. But Received: "
              "Input(CustomDistAliasProbs).numel() = %d, "
              "Attr(num_total_classes) = %d.",
              dist_alias_probs->numel(),
              num_total_classes));

      const float *probs_data = dist_probs->data<float>();
      const int *alias_data = dist_alias->data<int>();
      const float *alias_probs_data = dist_alias_probs->data<float>();
      sampler = new phi::math::CustomSampler(num_total_classes - 1,
                                             probs_data,
                                             alias_data,
                                             alias_probs_data,
                                             seed);
      break;
    }
    default: {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported SamplerType. SamplerType should be 0: Uniform, "
          "1: LogUniform or 2: CustomDist. Received SamplerType: %d",
          sampler_type));
    }
  }

  //    T b = 1. / num_total_classes * num_neg_samples;
  phi::DenseTensor sample_grad;  // tmp tensor
  sample_grad.Resize(sample_labels->dims());
  T *sample_grad_data = dev_ctx.template Alloc<T>(&sample_grad);

  // backward cost
  for (int64_t i = 0; i < sample_labels->numel(); ++i) {
    int64_t label_idx = i % sample_labels->dims()[1];
    int64_t sample_idx = i / sample_labels->dims()[1];
    float b = sampler->Probability(sample_labels_data[i]) * num_neg_samples;
    T o = sample_out_data[i];
    T w = sample_weight == nullptr ? 1 : sample_weight_data[sample_idx];
    sample_grad_data[i] = label_idx < num_true_class
                              ? w * (b / (o + b)) * (o - 1)
                              : w * (o * (1 - o) / (o + b));
    sample_grad_data[i] *= d_out_data[sample_idx];
  }

  // get d_bias
  auto d_bias = bias_grad;
  if (d_bias != nullptr) {
    T *d_bias_data = dev_ctx.template Alloc<T>(d_bias);
    std::fill(d_bias_data, d_bias_data + d_bias->numel(), 0.0);
    for (int64_t i = 0; i < sample_labels->numel(); ++i) {
      d_bias_data[sample_labels_data[i]] += sample_grad_data[i];
    }
  }

  if (!is_sparse) {
    // get d_w
    auto d_w = weight_grad;
    if (d_w != nullptr) {
      auto d_w_data = dev_ctx.template Alloc<T>(d_w);
      std::fill(d_w_data, d_w_data + d_w->numel(), 0.0);
      auto d_w_matrix = EigenMatrix<T>::From(*d_w);
      auto x_matrix = EigenMatrix<T>::From(input_in);
      for (int64_t i = 0; i < sample_labels->numel(); ++i) {
        d_w_matrix.chip(sample_labels_data[i], 0) +=
            x_matrix.chip(static_cast<int>(i / sample_labels->dims()[1]), 0) *
            sample_grad_data[i];
      }
    }
  } else {
    PADDLE_THROW(
        common::errors::InvalidArgument("The parameter weight_grad of a NCE_OP "
                                        "must be SelectedRows"));
  }

  // get d_x
  auto d_x = input_grad;
  if (d_x != nullptr) {
    auto *d_x_data = dev_ctx.template Alloc<T>(d_x);
    std::fill(d_x_data, d_x_data + d_x->numel(), 0.0);
    auto d_x_matrix = EigenMatrix<T>::From(*d_x);
    auto w_matrix = EigenMatrix<T>::From(weight_in);
    for (int64_t i = 0; i < sample_labels->numel(); ++i) {
      d_x_matrix.chip(static_cast<int>(i / sample_labels->dims()[1]), 0) +=
          w_matrix.chip(sample_labels_data[i], 0) * sample_grad_data[i];
    }
  }

  delete sampler;
}

}  // namespace phi

PD_REGISTER_KERNEL(
    nce_grad, CPU, ALL_LAYOUT, phi::NCEGradKernel, float, double) {}
