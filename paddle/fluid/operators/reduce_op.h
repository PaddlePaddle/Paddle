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

#include <vector>
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

#define HANDLE_DIM(NDIM, RDIM)          \
  if (ndim == NDIM && rdim == RDIM) {   \
    ReduceCompute<NDIM, RDIM>(context); \
  }

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
      int ndim = context.Input<Tensor>("X")->dims().size();
      int rdim = context.Attr<std::vector<int>>("dim").size();
      HANDLE_DIM(6, 5);
      HANDLE_DIM(6, 4);
      HANDLE_DIM(6, 3);
      HANDLE_DIM(6, 2);
      HANDLE_DIM(6, 1);
      HANDLE_DIM(5, 4);
      HANDLE_DIM(5, 3);
      HANDLE_DIM(5, 2);
      HANDLE_DIM(5, 1);
      HANDLE_DIM(4, 3);
      HANDLE_DIM(4, 2);
      HANDLE_DIM(4, 1);
      HANDLE_DIM(3, 2);
      HANDLE_DIM(3, 1);
      HANDLE_DIM(2, 1);
      HANDLE_DIM(1, 1);
    }
  }

 private:
  template <size_t D, size_t R_D>
  void ReduceCompute(const framework::ExecutionContext& context) const {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<T>(context.GetPlace());

    auto x = EigenTensor<T, D>::From(*input);
    auto x_rank = static_cast<int>(x.dimensions().size());
    auto dims = context.Attr<std::vector<int>>("dim");
    auto reduce_dim = Eigen::array<int, R_D>();
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] < 0) dims[i] = x_rank + dims[i];
      reduce_dim[i] = dims[i];
    }
    // construct the squeezed output tensor
    bool keep_dim = context.Attr<bool>("keep_dim");
    DDim out_dims = output->dims();
    if (keep_dim && x_rank > 1) {
      const int kDelFlag = -2;
      auto dims_vector = vectorize(out_dims);
      for (size_t i = 0; i < dims.size(); ++i) {
        dims_vector[dims[i]] = kDelFlag;
      }
      dims_vector.erase(
          remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
          dims_vector.end());
      out_dims = framework::make_ddim(dims_vector);
    }
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    Functor functor;

    if (D == 1) {
      auto out = EigenScalar<T>::From(*output);
      functor(place, &x, &out, reduce_dim);
    } else {
      auto out = EigenTensor<T, (D - R_D)>::From(*output, out_dims);
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
    auto dims = context.Attr<std::vector<int>>("dim");
    auto x_dims = input0->dims();
    auto reduced_dims_v = vectorize(x_dims);
    Eigen::array<int, D> broadcast_dim;
    for (size_t i = 0; i < D; ++i) broadcast_dim[i] = 1;

    int broad_cats_times = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] < 0) dims[i] = x_rank + dims[i];
      reduced_dims_v[dims[i]] = 1;
      broadcast_dim[dims[i]] = x_dims[dims[i]];
      broad_cats_times *= x_dims[dims[i]];
    }
    auto reduced_dims = framework::make_ddim(reduced_dims_v);
    auto x_reduce = EigenTensor<T, D>::From(*input1, reduced_dims);
    auto x_reduce_grad = EigenTensor<T, D>::From(*input2, reduced_dims);

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    Functor functor;
    functor(place, &x, &x_reduce, &x_grad, &x_reduce_grad, broadcast_dim,
            broad_cats_times);
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
