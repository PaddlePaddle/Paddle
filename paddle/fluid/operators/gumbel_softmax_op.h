/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D>;

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

static inline int SizeToAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeFromAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeOutAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename DeviceContext, typename T, int64_t Rank>
struct ArgMaxFunctor {
  void operator()(const DeviceContext& ctx, const Tensor& in,
                  Tensor* index_tensor, const int64_t& axis) {
    auto in_eigen = EigenTensor<T, Rank>::From(in, in.dims());
    auto index_eigen = EigenTensor<int, Rank - 1>::From(*index_tensor);
    index_eigen = in_eigen.argmax(axis).template cast<int>();
  }
};
template <typename DeviceContext, typename T>
struct GumbleNoiseGenerator;

template <typename DeviceContext, typename T>
struct OneHotGenerator;

template <typename T>
struct GumbleNoiseGenerator<platform::CPUDeviceContext, T> {
  static void Transform(const platform::CPUDeviceContext& context,
                        const T* input_data, T* output_data, int size_to_axis,
                        int size_from_axis, const float temperature) {
    // generate uniform random number
    const int size = size_to_axis * size_from_axis;
    std::uniform_real_distribution<T> dist(0.00001, 1);
    auto engine = paddle::framework::GetCPURandomEngine(0);
    Tensor random_tensor;
    auto* random_data =
        random_tensor.mutable_data<T>({size}, platform::CPUPlace());
    for (int64_t i = 0; i < size; ++i) {
      random_data[i] = dist(*engine);
    }

    // generate gumbel noise
    framework::DDim dim_2d{size_to_axis, size_from_axis};
    auto gumbel_noise_eigen = EigenMatrix<T>::From(random_tensor, dim_2d);
    gumbel_noise_eigen = -(((-(gumbel_noise_eigen.log())).log()));

    // add noise
    for (int64_t i = 0; i < size_to_axis * size_from_axis; i++) {
      output_data[i] = (input_data[i] + random_data[i]) / temperature;
    }
  }
};
template <typename T>
struct OneHotGenerator<platform::CPUDeviceContext, T> {
  static void Transform(const platform::CPUDeviceContext& context,
                        const Tensor& X, Tensor* Out, int axis) {
    Tensor index;
    std::vector<int> index_dim;
    const auto rank = X.dims().size();
    const int size_to_axis = SizeToAxis(axis, X.dims());
    const int size_from_axis = SizeFromAxis(axis, X.dims());
    const int size_out_axis = SizeOutAxis(axis, X.dims());

    for (int i = 0; i < X.dims().size(); i++) {
      if (i != axis) index_dim.push_back(X.dims().Get()[i]);
    }
    DDim index_ddim(index_dim.data(), rank - 1);
    index.Resize(index_ddim);
    auto* index_data = index.mutable_data<int>(context.GetPlace());

#define CALL_ARG_MINMAX_FUNCTOR(rank)                               \
  ArgMaxFunctor<platform::CPUDeviceContext, T, rank> functor##rank; \
  functor##rank(context, *Out, &index, axis);
    switch (Out->dims().size()) {
      case 1:
        CALL_ARG_MINMAX_FUNCTOR(1);
        break;
      case 2:
        CALL_ARG_MINMAX_FUNCTOR(2);
        break;
      case 3:
        CALL_ARG_MINMAX_FUNCTOR(3);
        break;
      case 4:
        CALL_ARG_MINMAX_FUNCTOR(4);
        break;
      case 5:
        CALL_ARG_MINMAX_FUNCTOR(5);
        break;
      case 6:
        CALL_ARG_MINMAX_FUNCTOR(6);
        break;
      default:
        PADDLE_ENFORCE_LE(Out->dims().size(), 6,
                          platform::errors::InvalidArgument(
                              "gumbel_softmax operator doesn't supports "
                              "tensors whose ranks are greater "
                              "than 6 in CPU mode."));
        break;
#undef CALL_ARG_MINMAX_FUNCTOR
    }

    pten::funcs::set_constant(context, Out, 0.0);
    for (int i = 0; i < size_to_axis; i++) {
      for (int j = 0; j < size_out_axis; j++) {
        *(Out->data<T>() + i * size_from_axis + j +
          index_data[i * size_out_axis + j] * size_out_axis) = 1.0;
      }
    }
  }
};

template <typename DeviceContext, typename T>
class GumbelSoftmaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");
    const int rank = X->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = X->dims()[axis];
    const bool is_hard = context.Attr<bool>("hard");
    const float temperature = context.Attr<float>("temperature");
    PADDLE_ENFORCE_GT(temperature, 0,
                      platform::errors::InvalidArgument(
                          "The temperature must be greater than 0. But "
                          "received temperature = %f",
                          temperature));

    // allocate memory on device.
    Out->mutable_data<T>(context.GetPlace());
    if (Out->numel() == 0) {
      return;
    }

    const int size_to_axis = SizeToAxis(axis, X->dims());
    const int size_from_axis = SizeFromAxis(axis, X->dims());
    Tensor X_noise_2d, Out_2d;
    X_noise_2d.Resize({size_to_axis, size_from_axis});
    Out_2d.ShareDataWith(*Out).Resize({size_to_axis, size_from_axis});

    // generate gumbel noise and add it to X
    auto* x_noise_data = X_noise_2d.mutable_data<T>(context.GetPlace());
    GumbleNoiseGenerator<DeviceContext, T>::Transform(
        context.template device_context<DeviceContext>(), X->data<T>(),
        x_noise_data, size_to_axis, size_from_axis, temperature);

#ifdef PADDLE_ON_INFERENCE
    math::SoftmaxFunctor<DeviceContext, T, true>()(
        context.template device_context<DeviceContext>(), axis_dim, &X_noise_2d,
        &Out_2d);
#else
    math::SoftmaxFunctor<DeviceContext, T, false>()(
        context.template device_context<DeviceContext>(), axis_dim, &X_noise_2d,
        &Out_2d);
#endif

    if (is_hard) {
      OneHotGenerator<DeviceContext, T>::Transform(
          context.template device_context<DeviceContext>(), *X, Out, axis);
    }
  }
};

template <typename DeviceContext, typename T>
class GumbelSoftmaxGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* Out = context.Input<Tensor>("Out");
    auto* dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dX = context.Output<Tensor>(framework::GradVarName("X"));
    const int rank = dX->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = dX->dims()[axis];
    // allocate memory on device.
    dX->mutable_data<T>(context.GetPlace());
    if (dX->numel() == 0) {
      return;
    }

    const int size_to_axis = SizeToAxis(axis, dX->dims());
    const int size_from_axis = SizeFromAxis(axis, dX->dims());
    Tensor dX_2d, Out_2d, dOut_2d;
    dX_2d.ShareDataWith(*dX).Resize({size_to_axis, size_from_axis});
    Out_2d.ShareDataWith(*Out).Resize({size_to_axis, size_from_axis});
    dOut_2d.ShareDataWith(*dOut).Resize({size_to_axis, size_from_axis});
    math::SoftmaxGradFunctor<DeviceContext, T>()(
        context.template device_context<DeviceContext>(), axis_dim, &Out_2d,
        &dOut_2d, &dX_2d);
  }
};

}  // namespace operators
}  // namespace paddle
