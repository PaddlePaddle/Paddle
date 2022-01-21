/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <math.h>
#include <iterator>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/operators/math/sampler.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using Sampler = math::Sampler;
using DDim = framework::DDim;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
void PrepareSamples(const framework::ExecutionContext &context,
                    Sampler *sampler, Tensor *sample_labels) {
  auto label = context.Input<Tensor>("Label");
  const int64_t *label_data = label->data<int64_t>();
  auto label_dims = label->dims();
  // for unitest
  std::vector<int> custom_neg_classes =
      context.Attr<std::vector<int>>("custom_neg_classes");

  auto sample_labels_dims = sample_labels->dims();
  int64_t *sample_labels_data =
      sample_labels->mutable_data<int64_t>(context.GetPlace());

  int num_label = label_dims.size() == 2 ? label_dims[1] : 1;
  int index = 0;
  for (int64_t i = 0; i < label_dims[0]; ++i) {
    int j = 0;
    for (; j < num_label; ++j) {
      sample_labels_data[index++] = label_data[i * num_label + j];
    }
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

template <typename DeviceContext, typename T>
class NCEKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    int sampler_type = context.Attr<int>("sampler");
    int seed = context.Attr<int>("seed");
    int num_total_classes = context.Attr<int>("num_total_classes");
    int num_neg_samples = context.Attr<int>("num_neg_samples");
    bool is_test = context.Attr<bool>("is_test");

    Sampler *sampler;
    switch (sampler_type) {
      case 0: {
        sampler = new math::UniformSampler(num_total_classes - 1, seed);
        break;
      }
      case 1: {
        sampler = new math::LogUniformSampler(num_total_classes - 1, seed);
        break;
      }
      case 2: {
        auto dist_probs = context.Input<Tensor>("CustomDistProbs");
        auto dist_alias = context.Input<Tensor>("CustomDistAlias");
        auto dist_alias_probs = context.Input<Tensor>("CustomDistAliasProbs");

        PADDLE_ENFORCE_EQ(
            dist_probs->numel(), num_total_classes,
            platform::errors::InvalidArgument(
                "ShapeError: The number of elements in Input(CustomDistProbs) "
                "should be equal to the number of total classes. But Received: "
                "Input(CustomDistProbs).numel() = %d, Attr(num_total_classes) "
                "= %d.",
                dist_probs->numel(), num_total_classes));
        PADDLE_ENFORCE_EQ(
            dist_alias->numel(), num_total_classes,
            platform::errors::InvalidArgument(
                "ShapeError: The number of elements in Input(CustomDistAlias) "
                "should be equal to the number of total classes. But Received: "
                "Input(CustomDistAlias).numel() = %d, Attr(num_total_classes) "
                "= %d.",
                dist_alias->numel(), num_total_classes));
        PADDLE_ENFORCE_EQ(
            dist_alias_probs->numel(), num_total_classes,
            platform::errors::InvalidArgument(
                "ShapeError: The number of elements in "
                "Input(CustomDistAliasProbs) "
                "should be equal to the number of total classes. But Received: "
                "Input(CustomDistAliasProbs).numel() = %d, "
                "Attr(num_total_classes) = %d.",
                dist_alias_probs->numel(), num_total_classes));

        const float *probs_data = dist_probs->data<float>();
        const int *alias_data = dist_alias->data<int>();
        const float *alias_probs_data = dist_alias_probs->data<float>();
        sampler = new math::CustomSampler(num_total_classes - 1, probs_data,
                                          alias_data, alias_probs_data, seed);
        break;
      }
      default: {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupported SamplerType. SamplerType should be 0: Uniform, "
            "1: LogUniform or 2: CostumDist. Received SamplerType: %d",
            sampler_type));
      }
    }

    std::vector<int64_t> sample_out_dims;
    auto label = context.Input<Tensor>("Label");
    Tensor *sample_labels;
    Tensor *sample_out;
    Tensor sample_labels_tmp, sample_out_tmp;
    if (is_test) {
      // set dims of output(SampleOut)
      int num_true_classes = label->dims().size() == 2 ? label->dims()[1] : 1;
      sample_out_dims.push_back((context.Input<Tensor>("Input"))->dims()[0]);
      sample_out_dims.push_back(
          (num_true_classes == -1) ? -1 : (num_neg_samples + num_true_classes));

      sample_labels = &sample_labels_tmp;
      sample_labels->Resize(framework::make_ddim(sample_out_dims));

      sample_out = &sample_out_tmp;
      sample_out->Resize(framework::make_ddim(sample_out_dims));
    } else {
      sample_labels = context.Output<Tensor>("SampleLabels");
      sample_out = context.Output<Tensor>("SampleLogits");
    }

    PrepareSamples<DeviceContext, T>(context, sampler, sample_labels);
    const int64_t *sample_labels_data = sample_labels->data<int64_t>();

    for (int x = 0; x < sample_labels->numel(); x++) {
      PADDLE_ENFORCE_GE(sample_labels_data[x], 0,
                        platform::errors::InvalidArgument(
                            "ValueError: Every sample label should be "
                            "non-negative. But received: "
                            "Input(SampleLabels)[%d] = %d",
                            x, sample_labels_data[x]));
    }

    T *sample_out_data = sample_out->mutable_data<T>(context.GetPlace());
    auto sample_weight = context.Input<Tensor>("SampleWeight");
    const T *sample_weight_data = nullptr;
    if (sample_weight != nullptr) {
      sample_weight_data = sample_weight->data<T>();
    }
    auto out = context.Output<Tensor>("Cost");
    T *out_data = out->mutable_data<T>(context.GetPlace());
    int64_t num_true_class = 1;
    if (label != nullptr) {
      num_true_class = label->dims()[1];
    }
    int64_t sampled_labels_num = sample_labels->dims()[1];
    //    T b = 1. / num_total_classes * num_neg_samples;
    // forward bias
    auto bias = context.Input<Tensor>("Bias");
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
    auto input_mat = EigenMatrix<T>::From(*(context.Input<Tensor>("Input")));

    auto weight_mat = EigenMatrix<T>::From(*(context.Input<Tensor>("Weight")));
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
};

template <typename DeviceContext, typename T>
class NCEGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto d_out = context.Input<Tensor>(framework::GradVarName("Cost"));
    const T *d_out_data = d_out->data<T>();
    auto label = context.Input<Tensor>("Label");
    auto sample_out = context.Input<Tensor>("SampleLogits");
    const T *sample_out_data = sample_out->data<T>();
    auto sample_labels = context.Input<Tensor>("SampleLabels");
    const int64_t *sample_labels_data = sample_labels->data<int64_t>();
    auto sample_weight = context.Input<Tensor>("SampleWeight");
    const T *sample_weight_data = nullptr;
    if (sample_weight != nullptr) {
      sample_weight_data = sample_weight->data<T>();
    }
    int num_neg_samples = context.Attr<int>("num_neg_samples");
    int num_total_classes = context.Attr<int>("num_total_classes");
    int num_true_class = 1;
    if (label != nullptr) {
      num_true_class = label->dims()[1];
    }

    int sampler_type = context.Attr<int>("sampler");
    int seed = context.Attr<int>("seed");
    Sampler *sampler;
    switch (sampler_type) {
      case 0: {
        sampler = new math::UniformSampler(num_total_classes - 1, seed);
        break;
      }
      case 1: {
        sampler = new math::LogUniformSampler(num_total_classes - 1, seed);
        break;
      }
      case 2: {
        auto dist_probs = context.Input<Tensor>("CustomDistProbs");
        auto dist_alias = context.Input<Tensor>("CustomDistAlias");
        auto dist_alias_probs = context.Input<Tensor>("CustomDistAliasProbs");

        PADDLE_ENFORCE_EQ(
            dist_probs->numel(), num_total_classes,
            platform::errors::InvalidArgument(
                "ShapeError: The number of elements in Input(CustomDistProbs) "
                "should be equal to the number of total classes. But Received: "
                "Input(CustomDistProbs).numel() = %d, Attr(num_total_classes) "
                "= %d.",
                dist_probs->numel(), num_total_classes));
        PADDLE_ENFORCE_EQ(
            dist_alias->numel(), num_total_classes,
            platform::errors::InvalidArgument(
                "ShapeError: The number of elements in Input(CustomDistAlias) "
                "should be equal to the number of total classes. But Received: "
                "Input(CustomDistAlias).numel() = %d, Attr(num_total_classes) "
                "= %d.",
                dist_alias->numel(), num_total_classes));
        PADDLE_ENFORCE_EQ(
            dist_alias_probs->numel(), num_total_classes,
            platform::errors::InvalidArgument(
                "ShapeError: The number of elements in "
                "Input(CustomDistAliasProbs) "
                "should be equal to the number of total classes. But Received: "
                "Input(CustomDistAliasProbs).numel() = %d, "
                "Attr(num_total_classes) = %d.",
                dist_alias_probs->numel(), num_total_classes));

        const float *probs_data = dist_probs->data<float>();
        const int *alias_data = dist_alias->data<int>();
        const float *alias_probs_data = dist_alias_probs->data<float>();
        sampler = new math::CustomSampler(num_total_classes - 1, probs_data,
                                          alias_data, alias_probs_data, seed);
        break;
      }
      default: {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupported SamplerType. SamplerType should be 0: Uniform, "
            "1: LogUniform or 2: CostumDist. Received SamplerType: %d",
            sampler_type));
      }
    }

    //    T b = 1. / num_total_classes * num_neg_samples;
    Tensor sample_grad;  // tmp tensor
    T *sample_grad_data =
        sample_grad.mutable_data<T>(sample_labels->dims(), context.GetPlace());
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
    auto d_bias = context.Output<Tensor>(framework::GradVarName("Bias"));
    if (d_bias != nullptr) {
      T *d_bias_data = d_bias->mutable_data<T>(context.GetPlace());
      std::fill(d_bias_data, d_bias_data + d_bias->numel(), 0.0);
      for (int64_t i = 0; i < sample_labels->numel(); ++i) {
        d_bias_data[sample_labels_data[i]] += sample_grad_data[i];
      }
    }

    bool is_sparse = context.Attr<bool>("is_sparse");

    if (!is_sparse) {
      // get d_w
      auto d_w = context.Output<Tensor>(framework::GradVarName("Weight"));
      if (d_w != nullptr) {
        auto d_w_data = d_w->mutable_data<T>(context.GetPlace());
        std::fill(d_w_data, d_w_data + d_w->numel(), 0.0);
        auto d_w_matrix = EigenMatrix<T>::From(*d_w);
        auto x_matrix = EigenMatrix<T>::From(*(context.Input<Tensor>("Input")));
        for (int64_t i = 0; i < sample_labels->numel(); ++i) {
          d_w_matrix.chip(sample_labels_data[i], 0) +=
              x_matrix.chip(static_cast<int>(i / sample_labels->dims()[1]), 0) *
              sample_grad_data[i];
        }
      }
    } else {
      std::vector<int64_t> labels;
      for (int64_t i = 0; i < sample_labels->numel(); ++i) {
        labels.push_back(sample_labels_data[i]);
      }
      std::set<T> st(labels.begin(), labels.end());
      labels.assign(st.begin(), st.end());

      auto *table_var = context.InputVar("Weight");
      DDim table_dim;
      if (table_var->IsType<LoDTensor>()) {
        table_dim = context.Input<LoDTensor>("Weight")->dims();
      } else if (table_var->IsType<SelectedRows>()) {
        auto *table_t = context.Input<SelectedRows>("Weight");
        table_dim = table_t->value().dims();
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The parameter Weight of a NCE_OP "
            "must be either LoDTensor or SelectedRows"));
      }

      auto d_w = context.Output<SelectedRows>(framework::GradVarName("Weight"));

      d_w->set_rows(labels);
      d_w->set_height(table_dim[0]);

      auto *d_table_value = d_w->mutable_value();
      d_table_value->Resize(
          {static_cast<int64_t>(labels.size()), table_dim[1]});
      auto d_w_data = d_table_value->mutable_data<T>(context.GetPlace());
      std::fill(d_w_data, d_w_data + d_table_value->numel(), 0.0);

      auto d_w_matrix = EigenMatrix<T>::From(*d_table_value);
      auto x_matrix = EigenMatrix<T>::From(*(context.Input<Tensor>("Input")));
      for (int64_t i = 0; i < sample_labels->numel(); ++i) {
        d_w_matrix.chip(d_w->Index(sample_labels_data[i]), 0) +=
            x_matrix.chip(static_cast<int>(i / sample_labels->dims()[1]), 0) *
            sample_grad_data[i];
      }
    }

    // get d_x
    auto d_x = context.Output<Tensor>(framework::GradVarName("Input"));
    if (d_x != nullptr) {
      auto *d_x_data = d_x->mutable_data<T>(context.GetPlace());
      std::fill(d_x_data, d_x_data + d_x->numel(), 0.0);
      auto d_x_matrix = EigenMatrix<T>::From(*d_x);
      auto w_matrix = EigenMatrix<T>::From(*(context.Input<Tensor>("Weight")));
      for (int64_t i = 0; i < sample_labels->numel(); ++i) {
        d_x_matrix.chip(static_cast<int>(i / sample_labels->dims()[1]), 0) +=
            w_matrix.chip(sample_labels_data[i], 0) * sample_grad_data[i];
      }
    }

    delete sampler;
  }
};
}  // namespace operators
}  // namespace paddle
