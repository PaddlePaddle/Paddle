/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
using Tensor = phi::DenseTensor;

template <typename T,
          int D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

template <typename T>
class MeanIoUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& place =
        *ctx.template device_context<phi::CPUContext>().eigen_device();
    // get input and output tensor
    auto* predictions = ctx.Input<phi::DenseTensor>("Predictions");
    auto* labels = ctx.Input<phi::DenseTensor>("Labels");
    auto* out_mean_iou = ctx.Output<phi::DenseTensor>("OutMeanIou");
    auto* out_wrong = ctx.Output<phi::DenseTensor>("OutWrong");
    auto* out_correct = ctx.Output<phi::DenseTensor>("OutCorrect");
    int num_classes = static_cast<int>(ctx.Attr<int>("num_classes"));

    // get data ptr
    const T* predictions_data = predictions->data<T>();
    const T* labels_data = labels->data<T>();
    float* out_mean_iou_data =
        out_mean_iou->mutable_data<float>(ctx.GetPlace());
    int* out_wrong_data = out_wrong->mutable_data<int>(ctx.GetPlace());
    int* out_correct_data = out_correct->mutable_data<int>(ctx.GetPlace());

    // get eigen tensor
    auto out_mean_iou_t = EigenTensor<float, 1>::From(*out_mean_iou);
    auto out_wrong_t = EigenTensor<int, 1>::From(*out_wrong);
    auto out_correct_t = EigenTensor<int, 1>::From(*out_correct);

    // Tmp tensor
    Tensor denominator;
    Tensor valid_count;
    Tensor iou_sum;

    // get data ptr of tmp tensor
    int* denominator_data = denominator.mutable_data<int>(
        {static_cast<int64_t>(num_classes)}, ctx.GetPlace());
    int* valid_count_data = valid_count.mutable_data<int>({1}, ctx.GetPlace());
    float* iou_sum_data = iou_sum.mutable_data<float>({1}, ctx.GetPlace());

    // get eigen tensor of tmp tensor
    auto denominator_t = EigenTensor<int, 1>::From(denominator);
    auto valid_count_t = EigenTensor<int, 1>::From(valid_count);
    auto iou_sum_t = EigenTensor<float, 1>::From(iou_sum);

    // init out_wrong, out_correct and out_mean_iou
    out_wrong_t = out_wrong_t.constant(0);
    out_correct_t = out_correct_t.constant(0);
    out_mean_iou_t = out_mean_iou_t.constant(0);

    // collect pre wrong, correct and mean_iou
    auto in_mean_ious = ctx.MultiInput<phi::DenseTensor>("InMeanIou");
    for (size_t i = 0; i < in_mean_ious.size(); ++i) {
      out_mean_iou_t.device(place) +=
          EigenTensor<float, 1>::From(*in_mean_ious[i]);
    }
    auto in_wrongs = ctx.MultiInput<phi::DenseTensor>("InWrongs");
    for (size_t i = 0; i < in_wrongs.size(); ++i) {
      out_wrong_t.device(place) += EigenTensor<int, 1>::From(*in_wrongs[i]);
    }
    auto in_corrects = ctx.MultiInput<phi::DenseTensor>("InCorrects");
    for (size_t i = 0; i < in_corrects.size(); ++i) {
      out_correct_t.device(place) += EigenTensor<int, 1>::From(*in_corrects[i]);
    }

    // compute
    for (int64_t i = 0; i < predictions->numel(); ++i) {
      if (predictions_data[i] == labels_data[i]) {
        out_correct_data[predictions_data[i]] += 1;
      } else {
        out_wrong_data[labels_data[i]] += 1;
        out_wrong_data[predictions_data[i]] += 1;
      }
    }

    denominator_t = out_wrong_t + out_correct_t;
    valid_count_t =
        (denominator_t > denominator_t.constant(0.0f)).cast<int>().sum();

    for (int i = 0; i < num_classes; ++i) {
      if (denominator_data[i] == 0) {
        denominator_data[i] = 1;
      }
    }

    iou_sum_t =
        (out_correct_t.cast<float>() / denominator_t.cast<float>()).sum();
    out_mean_iou_data[0] += (iou_sum_data[0] / valid_count_data[0]);
  }
};

}  // namespace operators
}  // namespace paddle
