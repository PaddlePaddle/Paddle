/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/cross_entropy.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
namespace funcs {

using Tensor = phi::DenseTensor;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = phi::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct HardLabelCrossEntropyCPUFunctorImpl {
  HardLabelCrossEntropyCPUFunctorImpl(phi::DenseTensor* out,
                                      const phi::DenseTensor* prob,
                                      const phi::DenseTensor* labels,
                                      const int ignore_index,
                                      const int axis_dim)
      : out_(out),
        prob_(prob),
        labels_(labels),
        ignore_index_(ignore_index),
        axis_dim_(axis_dim) {}

  template <typename U>
  void apply() const {
    const int batch_size = prob_->dims()[0];
    const int num_classes = prob_->dims()[1];
    const int num_remain = num_classes / axis_dim_;

    const T* prob_data = prob_->template data<T>();
    T* loss_data = out_->template data<T>();

    const auto* label_data = labels_->template data<U>();
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < num_remain; j++) {
        int lbl = static_cast<int>(label_data[i * num_remain + j]);  // NOLINT
        if (lbl != ignore_index_) {
          PADDLE_ENFORCE_GE(lbl,
                            0,
                            common::errors::OutOfRange(
                                "label value should >= 0 when label "
                                "value(%f) not equal to ignore_index(%f)",
                                lbl,
                                ignore_index_));
          PADDLE_ENFORCE_LT(
              lbl,
              axis_dim_,
              common::errors::OutOfRange(
                  "label value should less than the shape of axis dimension "
                  "when label value(%f) not equal to ignore_index(%f), But "
                  "received label value as %ld and shape of axis dimension "
                  "is %d",
                  lbl,
                  ignore_index_,
                  lbl,
                  axis_dim_));
        }
        int index = i * num_classes + lbl * num_remain + j;
        int loss_idx = i * num_remain + j;
        loss_data[loss_idx] =
            lbl == ignore_index_
                ? 0
                : -phi::funcs::TolerableValue<T>()(std::log(prob_data[index]));
      }
    }
  }

 private:
  phi::DenseTensor* out_;
  const phi::DenseTensor* prob_;
  const phi::DenseTensor* labels_;
  const int ignore_index_;
  const int axis_dim_;
};

template <typename DeviceContext, typename T>
void CrossEntropyFunctor<DeviceContext, T>::operator()(
    const DeviceContext& ctx,
    phi::DenseTensor* out,
    const phi::DenseTensor* prob,
    const phi::DenseTensor* labels,
    const bool softLabel,
    const int ignore_index,
    const int axis_dim) {
  if (softLabel) {
    const int batch_size = static_cast<const int>(prob->dims()[0]);
    const int num_classes = static_cast<const int>(prob->dims()[1]);
    const int num_remain = num_classes / axis_dim;

    Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);
    auto in = EigenMatrix<T>::From(*prob);
    auto lbl = EigenMatrix<T>::From(*labels);
    auto loss = EigenMatrix<T>::From(*out);

    loss.device(*ctx.eigen_device()) =
        -((lbl * in.log().unaryExpr(phi::funcs::TolerableValue<T>()))
              .reshape(batch_axis_remain)
              .sum(Eigen::DSizes<int, 1>(1)));
  } else {
    HardLabelCrossEntropyCPUFunctorImpl<T> functor_impl(
        out, prob, labels, ignore_index, axis_dim);
    phi::VisitDataType(labels->dtype(), functor_impl);
  }
}

template class CrossEntropyFunctor<phi::CPUContext, float>;
template class CrossEntropyFunctor<phi::CPUContext, double>;

}  // namespace funcs
}  // namespace phi
