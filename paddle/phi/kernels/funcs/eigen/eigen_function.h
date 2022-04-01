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
#pragma once
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "unsupported/Eigen/CXX11/Tensor"

namespace phi {
namespace funcs {

template <typename EigenDevice, typename T, int Rank>
struct EigenBroadcast {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using InType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<const T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;
  static void Eval(const EigenDevice& dev,
                   OutType out,
                   InType in,
                   const Array& bcast);
  static void Eval(const EigenDevice& dev,
                   OutType32BitIndex out,
                   InType32BitIndex in,
                   const Array& bcast);
};

template <typename EigenDevice, typename T, int Rank>
struct EigenBroadcastGrad {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using Array2 = Eigen::DSizes<Eigen::DenseIndex, Rank * 2>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType out,
                   InType in,
                   const Array& reduce_dims,
                   const Array2& reshape_dims);
};

template <typename EigenDevice, typename T, int Rank>
struct EigenConstant {
  using Type = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev, Type out, const T value);
};

template <typename EigenDevice, typename T>
struct EigenSign {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev, OutType out, const InType& in);
};

template <typename EigenDevice, typename T, int Rank>
struct EigenReverse {
  using Array = Eigen::DSizes<bool, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType out,
                   const InType& in,
                   const Array& reverse);
};

template <typename EigenDevice, typename T>
struct EigenAdd {
  using InType = Eigen::TensorMap<Eigen::TensorFixedSize<const T,
                                                         Eigen::Sizes<>,
                                                         Eigen::RowMajor,
                                                         Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<Eigen::TensorFixedSize<T,
                                                          Eigen::Sizes<>,
                                                          Eigen::RowMajor,
                                                          Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType out,
                   const InType& in,
                   const T value);
};

template <typename EigenDevice, typename T>
struct EigenSub {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType out,
                   const InType& left,
                   const InType& right);
};

template <typename EigenDevice, typename T, int Rank>
struct EigenSlice {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using Array32Bit = Eigen::DSizes<int, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using InType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<const T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;
  static void Eval(const EigenDevice& dev,
                   OutType out,
                   const InType& in,
                   const Array& offsets,
                   const Array& extents);
  static void Eval(const EigenDevice& dev,
                   OutType32BitIndex out,
                   const InType32BitIndex& in,
                   const Array32Bit& offsets,
                   const Array32Bit& extents);
};

template <typename EigenDevice, typename T, int Rank>
struct EigenPad {
  using Array = std::array<std::pair<int64_t, int64_t>, Rank>;
  using Array32Bit = std::array<std::pair<int, int>, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using InType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<const T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;
  static void Eval(const EigenDevice& dev,
                   OutType out,
                   const InType& in,
                   const Array& padding,
                   const T value);
  static void Eval32(const EigenDevice& dev,
                     OutType32BitIndex out,
                     const InType32BitIndex& in,
                     const Array32Bit& padding,
                     const T value);
};

template <typename EigenDevice, typename T>
struct EigenScale {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType out,
                   const InType& in,
                   const T scale,
                   const T bias,
                   const bool bias_after_scale);
};

template <typename EigenDevice, typename T>
struct EigenErf {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev, OutType out, const InType& in);
};

template <typename EigenDevice, typename T>
struct EigenErfGrad {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType din,
                   const InType& in,
                   const InType& dout);
};

template <typename EigenDevice, typename T>
struct EigenRankLoss {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType out,
                   const InType& label,
                   const InType& left,
                   const InType& right);
};

template <typename EigenDevice, typename T>
struct EigenRankLossGrad {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void EvalLeft(const EigenDevice& dev,
                       OutType dleft,
                       const InType& dout,
                       const InType& label,
                       const InType& left,
                       const InType& right);
  static void EvalRight(const EigenDevice& dev,
                        OutType dright,
                        const InType& dout,
                        const InType& label,
                        const InType& left,
                        const InType& right);
};

template <typename EigenDevice, typename T>
struct EigenLogLoss {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType out,
                   const InType& pred,
                   const InType& label,
                   const T& epsilon);
};

template <typename EigenDevice, typename T>
struct EigenLogLossGrad {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType dpred,
                   const InType& dloss,
                   const InType& pred,
                   const InType& label,
                   const T& epsilon);
};

template <typename EigenDevice, typename T>
struct EigenHingeLoss {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType loss,
                   const InType& pred,
                   const InType& label);
};

template <typename EigenDevice, typename T>
struct EigenHingeLossGrad {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType dpred,
                   const InType& dloss,
                   const InType& pred,
                   const InType& label);
};

template <typename EigenDevice, typename T>
struct EigenL1Norm {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<Eigen::TensorFixedSize<T,
                                                          Eigen::Sizes<>,
                                                          Eigen::RowMajor,
                                                          Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev, OutType out, const InType& in);
};

template <typename EigenDevice, typename T>
struct EigenL1NormGrad {
  using Array = Eigen::DSizes<Eigen::DenseIndex, 1>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev,
                   OutType din,
                   const InType& dout,
                   const InType& in,
                   const Array& bcast);
};

}  // namespace funcs
}  // namespace phi
