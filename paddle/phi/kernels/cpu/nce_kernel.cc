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
#include "paddle/utils/optional.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace phi {

using Sampler = phi::math::Sampler;

template <typename Context, typename T>
static void inline PrepareSamples(const Context &dev_ctx,
                                  Sampler *sampler,
                                  phi::DenseTensor *sample_labels,
                                  const phi::DenseTensor &label_in,
                                  const std::vector<int> &custom_neg_classes) {
  auto label = &label_in;
  const int64_t *label_data = label->data<int64_t>();
  auto label_dims = label->dims();

  auto sample_labels_dims = sample_labels->dims();
  int64_t *sample_labels_data = dev_ctx.template Alloc<int64_t>(sample_labels);

  int num_label = label_dims.size() == 2 ? label_dims[1] : 1;
  int index = 0;
  for (int64_t i = 0; i < label_dims[0]; ++i) {
    int j = 0;
    for (; j < num_label; ++j) {
      sample_labels_data[index++] = label_data[i * num_label + j];
    }
    // for unittest
    if (custom_neg_classes.size() > 0) {
      for (auto label : custom_neg_classes) {
        sample_labels_data[index++] = label;
      }
    } else {
      for (; j < sample_labels_dims[1]; ++j) {
        // TODO(wanghaoshuang): support more distribution sampling
        sample_labels_data[index++] = sampler->Sample();
      }
    }
  }
}

template <typename T, typename Context>
void NCEKernel(const Context &dev_ctx,
               const DenseTensor &input_in,
               const DenseTensor &label_in,
               const DenseTensor &weight_in,
               const paddle::optional<DenseTensor> &bias_in,
               const paddle::optional<DenseTensor> &sample_weight_in,
               const paddle::optional<DenseTensor> &custom_dist_probs,
               const paddle::optional<DenseTensor> &custom_dist_alias,
               const paddle::optional<DenseTensor> &custom_dist_alias_probs,
               int num_total_classes,
               const std::vector<int> &custom_neg_classes,
               int num_neg_samples,
               int sampler_in,
               int seed,
               bool is_sparse,
               bool remote_prefetch,
               bool is_test,
               DenseTensor *cost_out,
               DenseTensor *sample_logits_out,
               DenseTensor *sample_labels_out) {
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

  std::vector<int64_t> sample_out_dims;
  auto label = &label_in;
  phi::DenseTensor *sample_labels;
  phi::DenseTensor *sample_out;
  phi::DenseTensor sample_labels_tmp, sample_out_tmp;
  if (is_test) {
    // set dims of output(SampleOut)
    int num_true_classes = label->dims().size() == 2 ? label->dims()[1] : 1;
    sample_out_dims.push_back(input_in.dims()[0]);
    sample_out_dims.push_back(
        (num_true_classes == -1) ? -1 : (num_neg_samples + num_true_classes));

    sample_labels = &sample_labels_tmp;
    sample_labels->Resize(common::make_ddim(sample_out_dims));

    sample_out = &sample_out_tmp;
    sample_out->Resize(common::make_ddim(sample_out_dims));
  } else {
    sample_labels = sample_labels_out;
    sample_out = sample_logits_out;
  }

  PrepareSamples<Context, T>(
      dev_ctx, sampler, sample_labels, label_in, custom_neg_classes);
  const int64_t *sample_labels_data = sample_labels->data<int64_t>();

  for (int x = 0; x < sample_labels->numel(); x++) {
    PADDLE_ENFORCE_GE(sample_labels_data[x],
                      0,
                      common::errors::InvalidArgument(
                          "ValueError: Every sample label should be "
                          "non-negative. But received: "
                          "Input(SampleLabels)[%d] = %d",
                          x,
                          sample_labels_data[x]));
  }

  T *sample_out_data = dev_ctx.template Alloc<T>(sample_out);
  auto sample_weight = sample_weight_in.get_ptr();
  const T *sample_weight_data = nullptr;
  if (sample_weight != nullptr) {
    sample_weight_data = sample_weight->data<T>();
  }
  auto out = cost_out;
  T *out_data = dev_ctx.template Alloc<T>(out);
  int64_t num_true_class = 1;
  if (label != nullptr) {
    num_true_class = label->dims()[1];
  }
  int64_t sampled_labels_num = sample_labels->dims()[1];
  //    T b = 1. / num_total_classes * num_neg_samples;
  // forward bias
  auto bias = bias_in.get_ptr();
  if (bias != nullptr) {
    const T *bias_data = bias->data<T>();
    for (int64_t i = 0; i < sample_labels->numel(); ++i) {
      sample_out_data[i] = bias_data[sample_labels_data[i]];
    }
  } else {
    for (int64_t i = 0; i < sample_labels->numel(); ++i) {
      sample_out_data[i] = 0;
    }
  }
  // forward mul
  auto input_mat = EigenMatrix<T>::From(input_in);

  auto weight_mat = EigenMatrix<T>::From(weight_in);
  for (int64_t i = 0; i < sample_labels->numel(); ++i) {
    Eigen::Tensor<T, 0, Eigen::RowMajor, Eigen::DenseIndex> result =
        (input_mat.chip(static_cast<int>(i / sample_labels->dims()[1]), 0) *
         weight_mat.chip(sample_labels_data[i], 0))
            .sum();
    sample_out_data[i] += result(0);
    sample_out_data[i] = (1. / (1. + exp(-sample_out_data[i])));
  }

  // forward cost
  for (int64_t i = 0; i < sample_labels->dims()[0]; ++i) {
    out_data[i] = 0;
    T w = sample_weight == nullptr ? 1. : sample_weight_data[i];
    for (int64_t j = 0; j < sampled_labels_num; ++j) {
      int64_t target = sample_labels_data[i * sampled_labels_num + j];
      T o = sample_out_data[i * sampled_labels_num + j];
      float b = sampler->Probability(target) * num_neg_samples;
      T cost = (j < num_true_class) ? -log(o / (o + b)) : -log(b / (o + b));
      out_data[i] += w * cost;
    }
  }
  delete sampler;
}

}  // namespace phi

PD_REGISTER_KERNEL(nce, CPU, ALL_LAYOUT, phi::NCEKernel, float, double) {
  kernel->OutputAt(2).SetDataType(phi::DataType::INT64);
}
