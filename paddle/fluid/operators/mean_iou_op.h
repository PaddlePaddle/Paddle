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
#include <iostream>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
using cout = std::cout;
using endl = std::endl;
using Tensor = framework::Tensor;

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

template <typename DeviceContext, typename T>
struct GenConfusionMatrix {
  void operator()(const framework::ExecutionContext& ctx,
                  const int64_t num_classes, const int64_t count,
                  const T* predictions, const T* labels, const float* in_cm,
                  float* out_cm);
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

template <typename T>
void print_array(T* data, int n) {
  for (int i = 0; i < n; ++i) {
    cout << data[i] << ",";
  }
  cout << endl;
}

template <typename DeviceContext, typename T>
class MeanIoUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* predictions = ctx.Input<Tensor>("predictions");
    auto* labels = ctx.Input<Tensor>("labels");
    auto* in_cm = ctx.Input<Tensor>("in_confusion_matrix");
    auto* mean_iou = ctx.Output<Tensor>("mean_iou");
    auto* out_cm = ctx.Output<Tensor>("out_confusion_matrix");

    mean_iou->mutable_data<float>(ctx.GetPlace());
    const float* in_cm_data = in_cm->data<float>();
    float* out_cm_data = out_cm->mutable_data<float>(ctx.GetPlace());
    const T* predictions_data = predictions->data<T>();
    const T* labels_data = labels->data<T>();

    int64_t num_classes = static_cast<int64_t>(ctx.Attr<int>("num_classes"));

    // generate confusion matrix
    GenConfusionMatrix<DeviceContext, T> GCMFunctor;
    GCMFunctor(ctx, num_classes, predictions->numel(), predictions_data,
               labels_data, in_cm_data, out_cm_data);
    LOG(ERROR) << "in_cm_data: ";
    for (int i = 0; i < in_cm->numel(); ++i) {
      cout << in_cm_data[i] << ", ";
    }
    cout << endl;
    LOG(ERROR) << "out_cm_data: ";
    for (int i = 0; i < out_cm->numel(); ++i) {
      cout << out_cm_data[i] << ", ";
    }
    cout << endl;
    auto out_cm_t = EigenTensor<float, 2>::From(*out_cm);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    Tensor row_sum;
    float* row_sum_data =
        row_sum.mutable_data<float>({num_classes}, ctx.GetPlace());
    auto row_sum_t = EigenTensor<float, 1>::From(row_sum);
    Tensor col_sum;
    float* col_sum_data =
        col_sum.mutable_data<float>({num_classes}, ctx.GetPlace());
    auto col_sum_t = EigenTensor<float, 1>::From(col_sum);
    Tensor diag;
    float* diag_data = diag.mutable_data<float>({num_classes}, ctx.GetPlace());
    auto diag_t = EigenTensor<float, 1>::From(diag);
    Tensor denominator;
    float* denominator_data =
        denominator.mutable_data<float>({num_classes}, ctx.GetPlace());
    auto denominator_t = EigenTensor<float, 1>::From(denominator);

    Tensor valid_denominator;
    bool* valid_denominator_data =
        valid_denominator.mutable_data<bool>({num_classes}, ctx.GetPlace());
    auto valid_denominator_t = EigenTensor<bool, 1>::From(valid_denominator);

    Tensor valid_count;
    int64_t* valid_count_data =
        valid_count.mutable_data<int64_t>({1}, ctx.GetPlace());
    auto valid_count_t = EigenTensor<int64_t, 1>::From(valid_count);

    auto mean_iou_t = EigenTensor<float, 1>::From(*mean_iou);
    row_sum_t.device(place) = out_cm_t.sum(Eigen::array<int, 1>({{0}}));
    col_sum_t.device(place) = out_cm_t.sum(Eigen::array<int, 1>({{1}}));

    Diagonal<DeviceContext, float> diagonal;
    diagonal(ctx, num_classes, out_cm_data, diag_data);
    denominator_t.device(place) = row_sum_t + col_sum_t - diag_t;
    // denominator_t.constant(0.0f);

    valid_denominator_t.device(place) =
        (denominator_t > denominator_t.constant(0.0f));
    valid_count_t.device(place) = valid_denominator_t.cast<int64_t>().sum();

    Replace<DeviceContext, float> replace;
    replace(ctx, num_classes, denominator_data, 0.0f, 1.0f);

    mean_iou_t.device(place) =
        (diag_t / denominator_t).sum() / valid_count_t.cast<float>();

    LOG(ERROR) << "row_sum_data:";
    print_array<float>(row_sum_data, num_classes);
    LOG(ERROR) << "col_sum_data:";
    print_array<float>(col_sum_data, num_classes);
    LOG(ERROR) << "diag_data:";
    print_array<float>(diag_data, num_classes);
    LOG(ERROR) << "denominator_data:";
    print_array<float>(denominator_data, num_classes);
    LOG(ERROR) << "valid_denominator_data:";
    print_array<bool>(valid_denominator_data, num_classes);
    LOG(ERROR) << "valid_count_data:";
    print_array<int64_t>(valid_count_data, 1);
  }
};

}  // namespace operators
}  // namespace paddle
