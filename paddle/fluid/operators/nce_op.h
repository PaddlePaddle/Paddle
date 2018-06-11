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
#include <random>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "unsupported/Eigen/CXX11/Tensor"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
void PrepareSamples(const framework::ExecutionContext& context) {
  auto label = context.Input<Tensor>("Label");
  const int64_t* label_data = label->data<int64_t>();
  auto label_dims = label->dims();
  int num_total_classes = context.Attr<int>("num_total_classes");
  // for unitest
  std::vector<int> custom_neg_classes =
      context.Attr<std::vector<int>>("custom_neg_classes");
  // random machine
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> rand(0, num_total_classes - 1);

  auto sample_labels = context.Output<Tensor>("SampleLabels");
  auto sample_labels_dims = sample_labels->dims();
  int64_t* sample_labels_data =
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
        sample_labels_data[index++] = rand(rng);
      }
    }
  }
}

template <typename DeviceContext, typename T>
class NCEKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PrepareSamples<DeviceContext, T>(context);
    auto sample_labels = context.Output<Tensor>("SampleLabels");
    const int64_t* sample_labels_data = sample_labels->data<int64_t>();
    auto sample_out = context.Output<Tensor>("SampleLogits");
    T* sample_out_data = sample_out->mutable_data<T>(context.GetPlace());
    auto label = context.Input<Tensor>("Label");
    auto sample_weight = context.Input<Tensor>("SampleWeight");
    const T* sample_weight_data = nullptr;
    if (sample_weight != nullptr) {
      sample_weight_data = sample_weight->data<T>();
    }
    auto out = context.Output<Tensor>("Cost");
    T* out_data = out->mutable_data<T>(context.GetPlace());
    int num_neg_samples = context.Attr<int>("num_neg_samples");
    int num_total_classes = context.Attr<int>("num_total_classes");
    int64_t num_true_class = 1;
    if (label != nullptr) {
      num_true_class = label->dims()[1];
    }
    T b = 1. / num_total_classes * num_neg_samples;
    // forward bias
    auto bias = context.Input<Tensor>("Bias");
    if (bias != nullptr) {
      const T* bias_data = bias->data<T>();
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
      int64_t j = 0;
      out_data[i] = 0;
      T w = sample_weight == nullptr ? 1. : sample_weight_data[i];
      // for true classes
      for (; j < num_true_class; ++j) {
        T o = sample_out_data[i * sample_out->dims()[1] + j];
        T cost = -log(o / (o + b));
        out_data[i] += w * cost;
      }
      // for sampled neg classes
      for (; j < sample_labels->dims()[1]; ++j) {
        T o = sample_out_data[i * sample_out->dims()[1] + j];
        T cost = -log(b / (o + b));
        out_data[i] += w * cost;
      }
    }
  }
};

template <typename DeviceContext, typename T>
class NCEGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto d_out = context.Input<Tensor>(framework::GradVarName("Cost"));
    const T* d_out_data = d_out->data<T>();
    auto label = context.Input<Tensor>("Label");
    auto sample_out = context.Input<Tensor>("SampleLogits");
    const T* sample_out_data = sample_out->data<T>();
    auto sample_labels = context.Input<Tensor>("SampleLabels");
    const int64_t* sample_labels_data = sample_labels->data<int64_t>();
    auto sample_weight = context.Input<Tensor>("SampleWeight");
    const T* sample_weight_data = nullptr;
    if (sample_weight != nullptr) {
      sample_weight_data = sample_weight->data<T>();
    }
    int num_neg_samples = context.Attr<int>("num_neg_samples");
    int num_total_classes = context.Attr<int>("num_total_classes");
    int num_true_class = 1;
    if (label != nullptr) {
      num_true_class = label->dims()[1];
    }
    T b = 1. / num_total_classes * num_neg_samples;
    Tensor sample_grad;  // tmp tensor
    T* sample_grad_data =
        sample_grad.mutable_data<T>(sample_labels->dims(), context.GetPlace());
    // backward cost
    for (int64_t i = 0; i < sample_labels->numel(); ++i) {
      T o = sample_out_data[i];
      T w = sample_weight == nullptr
                ? 1
                : sample_weight_data[i / sample_labels->dims()[1]];
      sample_grad_data[i] = (i % sample_labels->dims()[1]) < num_true_class
                                ? w * (b / (o + b)) * (o - 1)
                                : w * (o * (1 - o) / (o + b));
      sample_grad_data[i] *= d_out_data[i / sample_labels->dims()[1]];
    }
    // get d_bias
    auto d_bias = context.Output<Tensor>(framework::GradVarName("Bias"));
    if (d_bias != nullptr) {
      T* d_bias_data = d_bias->mutable_data<T>(context.GetPlace());
      std::fill(d_bias_data, d_bias_data + d_bias->numel(), 0.0);
      for (int64_t i = 0; i < sample_labels->numel(); ++i) {
        d_bias_data[sample_labels_data[i]] += sample_grad_data[i];
      }
    }
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
    // get d_x
    auto d_x = context.Output<Tensor>(framework::GradVarName("Input"));
    if (d_x != nullptr) {
      auto* d_x_data = d_x->mutable_data<T>(context.GetPlace());
      std::fill(d_x_data, d_x_data + d_x->numel(), 0.0);
      auto d_x_matrix = EigenMatrix<T>::From(*d_x);
      auto w_matrix = EigenMatrix<T>::From(*(context.Input<Tensor>("Weight")));
      for (int64_t i = 0; i < sample_labels->numel(); ++i) {
        d_x_matrix.chip(static_cast<int>(i / sample_labels->dims()[1]), 0) +=
            w_matrix.chip(sample_labels_data[i], 0) * sample_grad_data[i];
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
