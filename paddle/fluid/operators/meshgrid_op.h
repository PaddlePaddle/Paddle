// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/platform/errors.h"

#define MAX_RANK_SUPPORTED 6

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MeshgridKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto ins = context.MultiInput<framework::Tensor>("X");
    auto rank = ins.size();
    switch (rank) {
      case 1:
        MeshgridForward<1>(context);
        break;
      case 2:
        MeshgridForward<2>(context);
        break;
      case 3:
        MeshgridForward<3>(context);
        break;
      case 4:
        MeshgridForward<4>(context);
        break;
      case 5:
        MeshgridForward<5>(context);
        break;
      case 6:
        MeshgridForward<6>(context);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Excepted Tensor numbers between 1 and 6, but only received d% .",
            rank));
    }
  }

 protected:
  template <int Rank>
  void MeshgridForward(const framework::ExecutionContext& context) const {}
};

template <typename DeviceContext, typename T>
class MeshgridGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto out_grad =
        context.MultiInput<framework::Tensor>(framework::GradVarName("Out"));
    int n = out_grad.size();
    switch (n) {
      case 1:
        MeshgridBackward<1>(context);
        break;
      case 2:
        MeshgridBackward<2>(context);
        break;
      case 3:
        MeshgridBackward<3>(context);
        break;
      case 4:
        MeshgridBackward<4>(context);
        break;
      case 5:
        MeshgridBackward<5>(context);
        break;
      case 6:
        MeshgridBackward<6>(context);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Excepted Tensor numbers between 1 and 6, but only received d% .",
            n));
    }
  }

 protected:
  template <int Rank>
  void MeshgridBackward(const framework::ExecutionContext& context) const {
    auto out_grad =
        context.MultiInput<framework::Tensor>(framework::GradVarName("Out"));
    auto ins = context.MultiInput<framework::Tensor>("X");
    auto outs =
        context.MultiOutput<framework::Tensor>(framework::GradVarName("X"));

    int n = out_grad.size();
    auto out_dims = out_grad[0]->dims();

    for (int i = 0; i < n; i++) {
      outs[i]->mutable_data<T>(context.GetPlace());
      auto out_grad_tmp = framework::EigenVector<T>::Flatten(*out_grad[i]);
      auto in_grad = framework::EigenVector<T>::Flatten(*outs[i]);

      std::vector<int> reduce_dims_vec;
      std::vector<int> reshape_dims_vec;
      for (int j = 0; j < n; j++) {
        reduce_dims_vec.push_back(reshape_dims_vec.size());
        if (j == i) {
          reshape_dims_vec.push_back(1);
          reshape_dims_vec.push_back(out_dims[j]);
        } else {
          reshape_dims_vec.push_back(out_dims[j]);
          reshape_dims_vec.push_back(1);
        }
      }

      Eigen::DSizes<Eigen::DenseIndex, Rank> reduce_dims;
      for (int k = 0; k < n; k++) {
        reduce_dims[k] = reduce_dims_vec[k];
      }

      Eigen::DSizes<Eigen::DenseIndex, Rank * 2> reshape_dims;
      for (int k = 0; k < n * 2; k++) {
        reshape_dims[k] = reshape_dims_vec[k];
      }

      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Rank>::Eval(
          place, in_grad, out_grad_tmp, reduce_dims, reshape_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle
