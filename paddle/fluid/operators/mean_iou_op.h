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
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

template <typename DeviceContext, typename T>
struct GenConfusionMatrix {
  void operator()(const framework::ExecutionContext& ctx,
                  const int64_t num_classes, const int64_t count,
                  const T* predictions, const T* labels, float* out_cm);
};

template <typename DeviceContext, typename T>
struct Replace {
  void operator()(const framework::ExecutionContext& ctx, const int64_t n,
                  T* data, T target, T value);
};

template <typename DeviceContext, typename T>
struct Diagonal {
  void operator()(const framework::ExecutionContext& ctx, int64_t n, T* data,
                  T* out);
};

template <typename DeviceContext, typename T>
class MeanIoUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    // get input and output tensor
    auto* predictions = ctx.Input<Tensor>("predictions");
    auto* labels = ctx.Input<Tensor>("labels");
    auto* mean_iou = ctx.Output<Tensor>("mean_iou");
    auto* out_cm = ctx.Output<Tensor>("out_confusion_matrix");
    int64_t num_classes = static_cast<int64_t>(ctx.Attr<int>("num_classes"));

    // get data ptr
    const T* predictions_data = predictions->data<T>();
    const T* labels_data = labels->data<T>();
    float* out_cm_data = out_cm->mutable_data<float>(ctx.GetPlace());
    mean_iou->mutable_data<float>(ctx.GetPlace());

    // get eigen tensor
    auto mean_iou_t = EigenTensor<float, 1>::From(*mean_iou);
    auto out_cm_t = EigenTensor<float, 2>::From(*out_cm);

    // Tmp tensor
    Tensor row_sum;
    Tensor col_sum;
    Tensor diag;
    Tensor denominator;
    Tensor valid_denominator;
    Tensor valid_count;

    // get data ptr of tmp tensor
    row_sum.mutable_data<float>({num_classes}, ctx.GetPlace());
    col_sum.mutable_data<float>({num_classes}, ctx.GetPlace());
    float* diag_data = diag.mutable_data<float>({num_classes}, ctx.GetPlace());
    float* denominator_data =
        denominator.mutable_data<float>({num_classes}, ctx.GetPlace());
    valid_denominator.mutable_data<bool>({num_classes}, ctx.GetPlace());
    valid_count.mutable_data<int64_t>({1}, ctx.GetPlace());

    // get eigen tensor of tmp tensor
    auto row_sum_t = EigenTensor<float, 1>::From(row_sum);
    auto col_sum_t = EigenTensor<float, 1>::From(col_sum);
    auto diag_t = EigenTensor<float, 1>::From(diag);
    auto denominator_t = EigenTensor<float, 1>::From(denominator);
    auto valid_denominator_t = EigenTensor<bool, 1>::From(valid_denominator);
    auto valid_count_t = EigenTensor<int64_t, 1>::From(valid_count);

    // init out_cm and mean_iou
    out_cm_t.device(place) = out_cm_t.constant(0.0f);
    mean_iou_t.device(place) = mean_iou_t.constant(0.0f);

    // collect pre mean_iou and confusion matrix
    auto in_mean_ious = ctx.MultiInput<Tensor>("in_mean_iou");
    for (size_t i = 0; i < in_mean_ious.size(); ++i) {
      mean_iou_t.device(place) += EigenTensor<float, 1>::From(*in_mean_ious[i]);
    }
    auto in_cms = ctx.MultiInput<Tensor>("in_confusion_matrix");
    for (size_t i = 0; i < in_cms.size(); ++i) {
      out_cm_t.device(place) += EigenTensor<float, 2>::From(*in_cms[i]);
    }

    // prepare functors
    GenConfusionMatrix<DeviceContext, T> GCMFunctor;
    Diagonal<DeviceContext, float> diagonal;
    Replace<DeviceContext, float> replace;

    // compute
    GCMFunctor(ctx, num_classes, predictions->numel(), predictions_data,
               labels_data, out_cm_data);
    row_sum_t.device(place) = out_cm_t.sum(Eigen::array<int, 1>({{0}}));
    col_sum_t.device(place) = out_cm_t.sum(Eigen::array<int, 1>({{1}}));
    diagonal(ctx, num_classes, out_cm_data, diag_data);
    denominator_t.device(place) = row_sum_t + col_sum_t - diag_t;
    valid_denominator_t.device(place) =
        (denominator_t > denominator_t.constant(0.0f));
    valid_count_t.device(place) = valid_denominator_t.cast<int64_t>().sum();
    replace(ctx, num_classes, denominator_data, 0.0f, 1.0f);
    mean_iou_t.device(place) +=
        (diag_t / denominator_t).sum() / valid_count_t.cast<float>();
  }
};

}  // namespace operators
}  // namespace paddle
