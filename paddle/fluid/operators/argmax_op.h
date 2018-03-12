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

struct ArgMaxFunctor {
  template <typename DeviceContext, typename X, typename Y>
  void operator()(const DeviceContext& place, X& x, Y& y, int idx) {
    y.device(place) = x.argmax(idx).template cast<int>();
  }
};

struct ArgMinFunctor {
  template <typename DeviceContext, typename X, typename Y>
  void operator()(const DeviceContext& place, X& x, Y& y, int idx) {
    y.device(place) = x.argmin(idx).template cast<int>();
  }
};

struct ArgExtremeGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X& x, Y& y, DX& dx, DY& dy,
                  const Dim& dim, int size) {
    auto equals = x == y.broadcast(dim);
    auto ones = dx.constant(1);
    auto zeros = dx.constant(0);
    dx.device(place) = dy.broadcast(dim) * equals.select(ones, zeros);
  }
};

template <typename DeviceContext, typename T, typename Functor>
class ArgExtremeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    if (reduce_all) {
      // Flatten and reduce 1-D tensor
      auto* input = context.Input<Tensor>("X");
      auto* output = context.Output<Tensor>("Out");
      output->mutable_data<int>(context.GetPlace());
      auto x = EigenVector<T>::Flatten(*input);
      auto out = EigenScalar<int>::From(*output);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      int reduce_dim = 0;
      Functor functor;
      functor(place, x, out, reduce_dim);
    } else {
      int rank = context.Input<Tensor>("X")->dims().size();
      switch (rank) {
        case 1:
          ArgExtremeCompute<1>(context);
          break;
        case 2:
          ArgExtremeCompute<2>(context);
          break;
        case 3:
          ArgExtremeCompute<3>(context);
          break;
        case 4:
          ArgExtremeCompute<4>(context);
          break;
        case 5:
          ArgExtremeCompute<5>(context);
          break;
        case 6:
          ArgExtremeCompute<6>(context);
          break;
      }
    }
  }

 private:
  template <size_t D>
  void ArgExtremeCompute(const framework::ExecutionContext& context) const {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<int>(context.GetPlace());

    auto x = EigenTensor<T, D>::From(*input);
    auto x_rank = static_cast<int>(x.dimensions().size());
    int dim = static_cast<int>(context.Attr<int>("dim"));
    if (dim < 0) dim = x_rank + dim;
    int reduce_dim = dim;
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
      auto out = EigenScalar<int>::From(*output);
      functor(place, x, out, reduce_dim);
    } else {
      auto out = EigenTensor<int, (D - 1)>::From(*output, dims);
      functor(place, x, out, reduce_dim);
    }
  }
};

template <typename DeviceContext, typename T, typename Functor>
class ArgExtremeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    if (reduce_all) {
      auto* input0 = context.Input<Tensor>("X");
      auto* input1 = context.Input<Tensor>("Out");
      auto* input2 = context.Input<Tensor>(framework::GradVarName("Out"));
      auto* output = context.Output<Tensor>(framework::GradVarName("X"));
      output->mutable_data<int>(context.GetPlace());
      auto x = EigenVector<T>::Flatten(*input0);
      auto x_argex = EigenVector<T>::From(*input1);
      auto x_argex_grad = EigenVector<T>::From(*input2);
      auto x_grad = EigenVector<T>::Flatten(*output);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto broadcast_dim =
          Eigen::array<int, 1>({{static_cast<int>(input0->numel())}});
      Functor functor;
      functor(place, x, x_argex, x_grad, x_argex_grad, broadcast_dim,
              broadcast_dim[0]);
    } else {
      int rank = context.Input<Tensor>("X")->dims().size();
      switch (rank) {
        case 1:
          ArgExtremeGradCompute<1>(context);
          break;
        case 2:
          ArgExtremeGradCompute<2>(context);
          break;
        case 3:
          ArgExtremeGradCompute<3>(context);
          break;
        case 4:
          ArgExtremeGradCompute<4>(context);
          break;
        case 5:
          ArgExtremeGradCompute<5>(context);
          break;
        case 6:
          ArgExtremeGradCompute<6>(context);
          break;
      }
    }
  }

 private:
  template <size_t D>
  void ArgExtremeGradCompute(const framework::ExecutionContext& context) const {
    auto* input0 = context.Input<Tensor>("X");
    auto* input1 = context.Input<Tensor>("Out");
    auto* input2 = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* output = context.Output<Tensor>(framework::GradVarName("X"));

    output->mutable_data<int>(context.GetPlace());
    auto x = EigenTensor<T, D>::From(*input0);
    auto x_grad = EigenTensor<T, D>::From(*output);
    auto x_rank = static_cast<int>(x.dimensions().size());
    int dim = static_cast<int>(context.Attr<int>("dim"));
    if (dim < 0) dim = x_rank + dim;
    DDim dims = input0->dims();
    dims[dim] = 1;
    auto x_argex = EigenTensor<T, D>::From(*input1, dims);
    auto x_argex_grad = EigenTensor<T, D>::From(*input2, dims);

    Eigen::array<int, D> broadcast_dim;
    for (size_t i = 0; i < D; ++i) broadcast_dim[i] = 1;
    broadcast_dim[dim] = input0->dims()[dim];
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    Functor functor;
    functor(place, x, x_argex, x_grad, x_argex_grad, broadcast_dim,
            broadcast_dim[dim]);
  }
};

}  // namespace operators
}  // namespace paddle

#define FOR_EACH_KERNEL_FUNCTOR(__macro)                 \
  __macro(argmax, ArgMaxFunctor, ArgExtremeGradFunctor); \
  __macro(argmin, ArgMinFunctor, ArgExtremeGradFunctor);
