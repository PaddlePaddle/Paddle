/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <random>
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/memory/memcpy.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
void PrepareSamples(const framework::ExecutionContext& context) {
  auto label = context.Input<Tensor>("Label");
  const T* label_data = label->data<T>();
  auto label_dims = label->dims();
  int num_classes = context.Attr<int>("num_classes");
  // random machine
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> rand(0, num_classes - 1);

  auto sample_labels = context.Output<Tensor>("SampleLabels");
  auto sample_labels_dims = sample_labels->dims();
  int* sample_labels_data =
      sample_labels->mutable_data<int>(context.GetPlace());

  int num_label = label_dims.size() == 2 ? label_dims[1] : 1;
  for (size_t i = 0; i < label_dims[0]; ++i) {
    int j = 0;
    for (; j < num_label; ++j) {
      sample_labels_data[sample_labels_dims[1] * i + j] =
          label_data[i * num_label + j];
    }
    for (; j < sample_labels_dims[1]; ++j) {
      int id = rand(rng);
      sample_labels_data[sample_labels_dims[1] * i + j] = id;
    }
  }
}

template <typename Place, typename T>
class NCEKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PrepareSamples<Place, T>(context);
    auto sample_labels = context.Output<Tensor>("SampleLabels");
    const int* sample_labels_data = sample_labels->data<int>();
    auto sample_out = context.Output<Tensor>("SampleLogits");
    T* sample_out_data = sample_out->mutable_data<T>(context.GetPlace());
    auto label = context.Input<Tensor>("Label");
    auto sample_weight = context.Input<Tensor>("SampleWeight");
    const T* sample_weight_data = nullptr;
    if (sample_weight != nullptr) {
      sample_weight_data = sample_weight->data<T>();
    }
    auto out = context.Output<Tensor>("Out");
    T* out_data = out->mutable_data<T>(context.GetPlace());
    int num_smalped_classes = context.Attr<int>("num_sampled_classes");
    int num_classes = context.Attr<int>("num_classes");
    int num_true_class = 1;
    if (label != nullptr) {
      num_true_class = label->dims()[1];
    }
    T b = 1. / num_classes * num_smalped_classes;

    // forward bias
    auto bias = context.Input<Tensor>("B");
    if (bias != nullptr) {
      const T* bias_data = bias->data<T>();
      for (size_t i = 0; i < sample_labels->numel(); ++i) {
        sample_out_data[i] = bias_data[sample_labels_data[i]];
      }
    } else {
      for (size_t i = 0; i < sample_labels->numel(); ++i) {
        sample_out_data[i] = 0;
      }
    }

    // forward mul
    auto input_mat = EigenMatrix<T>::From(*(context.Input<Tensor>("X")));
    auto weight_mat = EigenMatrix<T>::From(*(context.Input<Tensor>("W")));
    for (size_t i = 0; i < sample_labels->numel(); ++i) {
      // sample_out_data[i] += (input_mat.chip((int)(i /
      // sample_labels->dims()[1]), 0) * weight_mat.chip(sample_labels_data[i],
      // 0)).sum();
      Eigen::Tensor<float, 0, Eigen::RowMajor, Eigen::DenseIndex> result =
          (input_mat.chip((int)(i / sample_labels->dims()[1]), 0) *
           weight_mat.chip(sample_labels_data[i], 0))
              .sum();
      sample_out_data[i] += result(0);
      // activation_->forward
      sample_out_data[i] = (1 / 1 + (sample_out_data[i]));
    }

    // forward cost
    for (size_t i = 0; i < sample_labels->dims()[0]; ++i) {
      size_t j = 0;
      T w = sample_weight == nullptr ? 1 : sample_weight_data[i];
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

template <typename Place, typename T>
class NCEGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto label = context.Input<Tensor>("Label");
    auto sample_out = context.Input<Tensor>("SampleLogits");
    const T* sample_out_data = sample_out->data<T>();
    auto sample_labels = context.Input<Tensor>("SampleLabels");
    const int* sample_labels_data = sample_labels->data<int>();
    auto sample_weight = context.Input<Tensor>("SampleWeight");
    const T* sample_weight_data = nullptr;
    if (sample_weight != nullptr) {
      sample_weight_data = sample_weight->data<T>();
    }
    int num_smalped_classes = context.Attr<int>("num_sampled_classes");
    int num_classes = context.Attr<int>("num_classes");
    int num_true_class = 1;
    if (label != nullptr) {
      num_true_class = label->dims()[1];
    }
    T b = 1. / num_classes * num_smalped_classes;

    Tensor sample_grad;  // tmp tensor
    T* sample_grad_data =
        sample_grad.mutable_data<T>(sample_labels->dims(), context.GetPlace());

    // backward cost
    for (size_t i = 0; i < sample_labels->numel(); ++i) {
      T o = sample_out_data[i];
      T w = sample_weight == nullptr
                ? 1
                : sample_weight_data[i / sample_labels->dims()[1]];
      sample_grad_data[i] = (i % sample_labels->dims()[1]) < num_true_class
                                ? -w * b / (o * (o + b))
                                : w / (o + b);
      // sigmoid->backward
      sample_grad_data[i] =
          (o > 0) ? sample_grad_data[i] : ((o < 0) ? -sample_grad_data[i] : 0);
    }

    // get d_bias
    auto d_bias = context.Output<Tensor>(framework::GradVarName("B"));
    if (d_bias != nullptr) {
      T* d_bias_data = d_bias->mutable_data<T>(context.GetPlace());
      for (size_t i = 0; i < sample_labels->numel(); ++i) {
        d_bias_data[sample_labels_data[i]] += sample_grad_data[i];
      }
    }
    // get d_w
    auto d_w = context.Output<Tensor>(framework::GradVarName("W"));
    if (d_w != nullptr) {
      auto d_w_matrix = EigenMatrix<T>::From(*d_w);
      auto x_matrix = EigenMatrix<T>::From(*(context.Input<Tensor>("X")));
      for (size_t i = 0; i < sample_labels->numel(); ++i) {
        d_w_matrix.chip(sample_labels_data[i], 0) =
            x_matrix.chip((int)(i / sample_labels->dims()[1]), 0) *
            sample_grad_data[i];
      }
    }

    // get d_x
    auto d_x = context.Output<Tensor>(framework::GradVarName("X"));
    if (d_x != nullptr) {
      auto d_x_matrix = EigenMatrix<T>::From(*d_x);
      auto w_matrix = EigenMatrix<T>::From(*(context.Input<Tensor>("W")));
      for (size_t i = 0; i < sample_labels->numel(); ++i) {
        d_x_matrix.chip((int)(i / sample_labels->dims()[1]), 0) +=
            w_matrix.chip(sample_labels_data[i], 0) * sample_grad_data[i];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
