/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   You may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include <iostream>
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

template <typename Place, typename T>
class ExpandKernel : public framework::OpKernel {
 private:
  struct ExpandComputeVisitor : public boost::static_visitor<void> {
    explicit ExpandComputeVisitor(const framework::ExecutionContext& context)
        : context_(context) {}

    template <typename EigenDim>
    void operator()(const EigenDim& dim) const {
      auto* in0 = context_.Input<Tensor>("X");
      auto* out0 = context_.Output<Tensor>("Out");
      auto x = EigenTensor<T, EigenDim::count>::From(*in0);
      out0->template mutable_data<T>(context_.GetPlace());
      auto y = EigenTensor<T, EigenDim::count>::From(*out0);
      auto place = context_.GetEigenDevice<Place>();
      y.device(place) = x.broadcast(dim);
    }

    const framework::ExecutionContext& context_;
  };

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& expandTimes = context.Attr<std::vector<int>>("expandTimes");
    auto bcast_dim =
        framework::DDimToEigenDDim(framework::make_ddim(expandTimes));
    ExpandComputeVisitor visitor(context);
    framework::VisitEigenDDim(visitor, bcast_dim);
  }
};

template <typename Place, typename T>
class ExpandGradKernel : public framework::OpKernel {
 private:
  template <typename ReduceDim>
  struct ReshapeVisitor : public boost::static_visitor<void> {
    ReshapeVisitor(const framework::ExecutionContext& context,
                   const ReduceDim& reduce_dim)
        : context_(context), reduce_dim_(reduce_dim) {}

    template <typename ReshapeDim>
    typename std::enable_if<ReshapeDim::count >= ReduceDim::count, void>::type
    operator()(const ReshapeDim& reshape_dim) const {
      auto* in0 = context_.Input<Tensor>(framework::GradVarName("Out"));
      auto* out0 = context_.Output<Tensor>(framework::GradVarName("X"));
      auto x = EigenVector<T>::Flatten(*(context_.Input<Tensor>("X")));
      out0->template mutable_data<T>(context_.GetPlace());
      auto x_grad = EigenVector<T>::Flatten(*out0);
      auto out_grad = EigenVector<T>::Flatten(*in0);
      x_grad.device(context_.GetEigenDevice<Place>()) =
          out_grad.reshape(reshape_dim)
              .sum(reduce_dim_)
              .reshape(x.dimensions());
    }

    template <typename ReshapeDim>
        typename std::enable_if <
        ReshapeDim::count<ReduceDim::count, void>::type operator()(
            const ReshapeDim& reshape_dim) const {
      PADDLE_THROW("Reshape Dim %d is less than reduce dim %d",
                   ReshapeDim::count, ReduceDim::count);
    }

    const framework::ExecutionContext& context_;
    ReduceDim reduce_dim_;
  };

  struct ReduceVisitor : public boost::static_visitor<void> {
    ReduceVisitor(const framework::ExecutionContext& context,
                  const framework::EigenDDim& reshape_ddim)
        : context_(context), reshape_ddim_(reshape_ddim) {}

    template <typename ReduceDim>
    void operator()(const ReduceDim& reduce_dim) const {
      ReshapeVisitor<ReduceDim> visitor(context_, reduce_dim);
      boost::apply_visitor(visitor, reshape_ddim_);
    }

    const framework::ExecutionContext& context_;
    const framework::EigenDDim& reshape_ddim_;
  };

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<Tensor>("X");
    auto expand_times = context.Attr<std::vector<int>>("expandTimes");
    auto x_dims = in0->dims();
    std::vector<int> reshape_dims_vec;
    std::vector<int> reduce_dims_vec;
    for (size_t i = 0; i < expand_times.size(); ++i) {
      if (expand_times[i] == 1) {
        reshape_dims_vec.push_back(x_dims[i]);
      } else {
        if (x_dims[i] == 1) {
          reduce_dims_vec.push_back(reshape_dims_vec.size());
          reshape_dims_vec.push_back(expand_times[i]);
        } else {
          reduce_dims_vec.push_back(reshape_dims_vec.size());
          reshape_dims_vec.push_back(expand_times[i]);
          reshape_dims_vec.push_back(x_dims[i]);
        }
      }
    }

    // no need reduce, just copy
    if (reduce_dims_vec.size() == 0) {
      auto* in0 = context.Input<Tensor>(framework::GradVarName("Out"));
      auto* out0 = context.Output<Tensor>(framework::GradVarName("X"));
      out0->mutable_data<T>(context.GetPlace());
      if (platform::is_cpu_place(context.GetPlace())) {
        out0->CopyFrom<T>(*in0, platform::CPUPlace());
      } else {
        out0->CopyFrom<T>(*in0, platform::GPUPlace());
      }
    } else {
      auto reshape_ddim =
          framework::DDimToEigenDDim(framework::make_ddim(reshape_dims_vec));
      auto reduce_ddim =
          framework::DDimToEigenDDim(framework::make_ddim(reduce_dims_vec));

      ReduceVisitor visitor(context, reshape_ddim);
      boost::apply_visitor(visitor, reduce_ddim);
    }
  }
};

}  // namespace operators
}  // namespace paddle
