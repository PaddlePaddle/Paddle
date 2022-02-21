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
#include "paddle/pten/kernels/funcs/eigen/eigen_function.h"

namespace pten {
namespace funcs {

template <typename T>
struct EigenRankLoss<Eigen::GpuDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::GpuDevice& dev,
                   OutType out,
                   const InType& label,
                   const InType& left,
                   const InType& right) {
    out.device(dev) =
        (1.0f + (left - right).exp()).log() - label * (left - right);
  }
};

template <typename T>
struct EigenRankLossGrad<Eigen::GpuDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;

  static void EvalLeft(const Eigen::GpuDevice& dev,
                       OutType dleft,
                       const InType& dout,
                       const InType& label,
                       const InType& left,
                       const InType& right) {
    dleft.device(dev) = dout * (1.0f / (1.0f + (right - left).exp()) - label);
  }

  static void EvalRight(const Eigen::GpuDevice& dev,
                        OutType dright,
                        const InType& dout,
                        const InType& label,
                        const InType& left,
                        const InType& right) {
    dright.device(dev) = -dout * (1.0f / (1.0f + (right - left).exp()) - label);
  }
};

template struct EigenRankLoss<Eigen::GpuDevice, float>;
template struct EigenRankLossGrad<Eigen::GpuDevice, float>;

template <typename T>
struct EigenLogLoss<Eigen::GpuDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::GpuDevice& dev,
                   OutType out,
                   const InType& pred,
                   const InType& label,
                   const T& epsilon) {
    out.device(dev) = (-(label * (pred + epsilon).log()) -
                       ((static_cast<T>(1) - label) *
                        (static_cast<T>(1) - pred + epsilon).log()));
  }
};

template <typename T>
struct EigenLogLossGrad<Eigen::GpuDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::GpuDevice& dev,
                   OutType dpred,
                   const InType& dloss,
                   const InType& pred,
                   const InType& label,
                   const T& epsilon) {
    dpred.device(dev) =
        dloss *
        (-(label / (pred + epsilon)) +
         ((static_cast<T>(1) - label) / (static_cast<T>(1) - pred + epsilon)));
  }
};

template struct EigenLogLoss<Eigen::GpuDevice, float>;
template struct EigenLogLossGrad<Eigen::GpuDevice, float>;

template <typename T>
struct EigenHingeLoss<Eigen::GpuDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::GpuDevice& dev,
                   OutType loss,
                   const InType& pred,
                   const InType& label) {
    loss.device(dev) = (static_cast<T>(1) -
                        pred * (static_cast<T>(2) * label - static_cast<T>(1)))
                           .cwiseMax(static_cast<T>(0));
  }
};

template <typename T>
struct EigenHingeLossGrad<Eigen::GpuDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::GpuDevice& dev,
                   OutType dpred,
                   const InType& dloss,
                   const InType& pred,
                   const InType& label) {
    auto alt_labels = static_cast<T>(2) * label - static_cast<T>(1);
    dpred.device(dev) =
        dloss * ((pred * alt_labels) < static_cast<T>(1)).template cast<T>() *
        (-alt_labels);
  }
};

template struct EigenHingeLoss<Eigen::GpuDevice, float>;
template struct EigenHingeLossGrad<Eigen::GpuDevice, float>;

}  // namespace funcs
}  // namespace pten
