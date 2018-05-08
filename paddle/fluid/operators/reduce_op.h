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

#include "glog/logging.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

struct SumFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->sum(dim);
  }
};

struct SumGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    dx->device(place) = dy->broadcast(dim);
  }
};

struct MeanFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->mean(dim);
  }
};

struct MeanGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    dx->device(place) = dy->broadcast(dim) / dx->constant(size);
  }
};

struct MaxFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->maximum(dim);
  }
};

struct MinFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->minimum(dim);
  }
};

struct MaxOrMinGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    auto equals = (*x) == y->broadcast(dim);
    auto ones = dx->constant(1);
    auto zeros = dx->constant(0);
    // If there are multiple minimum or maximum elements, the subgradient of
    // each is the set [0, 1], and we pass gradient to all of them here.
    dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros);
  }
};

struct ProdFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->prod(dim);
  }
};

struct ProdGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    dx->device(place) = dy->broadcast(dim) * y->broadcast(dim) * x->inverse();
  }
};

template <typename DeviceContext, typename T, typename Functor>
class ReduceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    if (reduce_all) {
      // Flatten and reduce 1-D tensor
      auto* input = context.Input<Tensor>("X");
      auto* output = context.Output<Tensor>("Out");
      output->mutable_data<T>(context.GetPlace());
      auto x = EigenVector<T>::Flatten(*input);
      auto out = EigenScalar<T>::From(*output);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({{0}});
      Functor functor;
      functor(place, &x, &out, reduce_dim);
    } else {
      int rank = context.Input<Tensor>("X")->dims().size();
      switch (rank) {
        case 1:
          ReduceCompute<1>(context);
          break;
        case 2:
          ReduceCompute<2>(context);
          break;
        case 3:
          ReduceCompute<3>(context);
          break;
        case 4:
          ReduceCompute<4>(context);
          break;
        case 5:
          ReduceCompute<5>(context);
          break;
        case 6:
          ReduceCompute<6>(context);
          break;
      }
    }
  }

 private:
  template <size_t D>
  void ReduceCompute(const framework::ExecutionContext& context) const {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<T>(context.GetPlace());

    auto x = EigenTensor<T, D>::From(*input);
    auto x_rank = static_cast<int>(x.dimensions().size());
    int dim = static_cast<int>(context.Attr<int>("dim"));
    if (dim < 0) dim = x_rank + dim;
    auto reduce_dim = Eigen::array<int, 1>({{dim}});
    // construct the squeezed output tensor
    bool keep_dim = context.Attr<bool>("keep_dim");
    DDim dims = output->dims();
    auto dims_vector = vectorize(dims);
    if (keep_dim && x_rank > 1) {
      dims_vector.erase(dims_vector.begin() + dim);
      dims = framework::make_ddim(dims_vector);
    }

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    Functor functor;

    if (D == 1) {
      auto out = EigenScalar<T>::From(*output);
      functor(place, &x, &out, reduce_dim);
    } else {
      auto out = EigenTensor<T, (D - 1)>::From(*output, dims);
      functor(place, &x, &out, reduce_dim);
    }
  }
};

template <typename DeviceContext, typename T, typename Functor>
class ReduceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    if (reduce_all) {
      auto* input0 = context.Input<Tensor>("X");
      auto* input1 = context.Input<Tensor>("Out");
      auto* input2 = context.Input<Tensor>(framework::GradVarName("Out"));
      auto* output = context.Output<Tensor>(framework::GradVarName("X"));
      output->mutable_data<T>(context.GetPlace());
      auto x = EigenVector<T>::Flatten(*input0);
      auto x_reduce = EigenVector<T>::From(*input1);
      auto x_reduce_grad = EigenVector<T>::From(*input2);
      auto x_grad = EigenVector<T>::Flatten(*output);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto broadcast_dim =
          Eigen::array<int, 1>({{static_cast<int>(input0->numel())}});
      Functor functor;
      functor(place, &x, &x_reduce, &x_grad, &x_reduce_grad, broadcast_dim,
              broadcast_dim[0]);
    } else {
      int rank = context.Input<Tensor>("X")->dims().size();
      switch (rank) {
        case 1:
          ReduceGradCompute<1>(context);
          break;
        case 2:
          ReduceGradCompute<2>(context);
          break;
        case 3:
          ReduceGradCompute<3>(context);
          break;
        case 4:
          ReduceGradCompute<4>(context);
          break;
        case 5:
          ReduceGradCompute<5>(context);
          break;
        case 6:
          ReduceGradCompute<6>(context);
          break;
      }
    }
  }

 private:
  template <size_t D>
  void ReduceGradCompute(const framework::ExecutionContext& context) const {
    auto* input0 = context.Input<Tensor>("X");
    auto* input1 = context.Input<Tensor>("Out");
    auto* input2 = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* output = context.Output<Tensor>(framework::GradVarName("X"));

    output->mutable_data<T>(context.GetPlace());
    auto x = EigenTensor<T, D>::From(*input0);
    auto x_grad = EigenTensor<T, D>::From(*output);
    auto x_rank = static_cast<int>(x.dimensions().size());
    int dim = static_cast<int>(context.Attr<int>("dim"));
    if (dim < 0) dim = x_rank + dim;
    DDim dims = input0->dims();
    dims[dim] = 1;
    auto x_reduce = EigenTensor<T, D>::From(*input1, dims);
    auto x_reduce_grad = EigenTensor<T, D>::From(*input2, dims);

    Eigen::array<int, D> broadcast_dim;
    for (size_t i = 0; i < D; ++i) broadcast_dim[i] = 1;
    broadcast_dim[dim] = input0->dims()[dim];
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    Functor functor;
    functor(place, &x, &x_reduce, &x_grad, &x_reduce_grad, broadcast_dim,
            broadcast_dim[dim]);
  }
};

}  // namespace operators
}  // namespace paddle

#define FOR_EACH_KERNEL_FUNCTOR(__macro)                \
  __macro(reduce_sum, SumFunctor, SumGradFunctor);      \
  __macro(reduce_mean, MeanFunctor, MeanGradFunctor);   \
  __macro(reduce_max, MaxFunctor, MaxOrMinGradFunctor); \
  __macro(reduce_min, MinFunctor, MaxOrMinGradFunctor); \
  __macro(reduce_prod, ProdFunctor, ProdGradFunctor);
