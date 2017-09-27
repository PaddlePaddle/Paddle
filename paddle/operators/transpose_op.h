/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename Place, typename T, int Rank>
void EigenTranspose(const framework::ExecutionContext& context,
                    const framework::Tensor& in, framework::Tensor& out,
                    std::vector<int> axis) {
  Eigen::array<int, Rank> permute;
  for (int i = 0; i < Rank; i++) {
    permute[i] = axis[i];
  }
  auto in_dim = in.dims();
  auto out_dim = out.dims();

  auto eigen_in = framework::EigenTensor<T, Rank>::From(in);
  auto eigen_out = framework::EigenTensor<T, Rank>::From(out);
  auto& dev = context.GetEigenDevice<Place>();
  eigen_out.device(dev) = eigen_in.shuffle(permute);
}

template <typename Place, typename T>
class TransposeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    std::vector<int> axis = context.Attr<std::vector<int>>("axis");
    int ndims = axis.size();
    switch (ndims) {
      case 1:
        EigenTranspose<Place, T, 1>(context, *x, *out, axis);
        break;
      case 2:
        EigenTranspose<Place, T, 2>(context, *x, *out, axis);
        break;
      case 3:
        EigenTranspose<Place, T, 3>(context, *x, *out, axis);
        break;
      case 4:
        EigenTranspose<Place, T, 4>(context, *x, *out, axis);
        break;
      case 5:
        EigenTranspose<Place, T, 5>(context, *x, *out, axis);
        break;
      case 6:
        EigenTranspose<Place, T, 6>(context, *x, *out, axis);
        break;
      default:
        PADDLE_THROW("Tensors with rank at most 6 are supported");
    }
  }
};

template <typename Place, typename T>
class TransposeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    if (x_grad) {
      x_grad->mutable_data<T>(context.GetPlace());

      std::vector<int> axis = context.Attr<std::vector<int>>("axis");
      std::vector<int> reversed_axis(axis);

      for (size_t i = 0; i < axis.size(); i++) {
        reversed_axis[axis[i]] = i;
      }

      int ndims = axis.size();

      switch (ndims) {
        case 1:
          EigenTranspose<Place, T, 1>(context, *out_grad, *x_grad,
                                      reversed_axis);
          break;
        case 2:
          EigenTranspose<Place, T, 2>(context, *out_grad, *x_grad,
                                      reversed_axis);
          break;
        case 3:
          EigenTranspose<Place, T, 3>(context, *out_grad, *x_grad,
                                      reversed_axis);
          break;
        case 4:
          EigenTranspose<Place, T, 4>(context, *out_grad, *x_grad,
                                      reversed_axis);
          break;
        case 5:
          EigenTranspose<Place, T, 5>(context, *out_grad, *x_grad,
                                      reversed_axis);
          break;
        case 6:
          EigenTranspose<Place, T, 6>(context, *out_grad, *x_grad,
                                      reversed_axis);
          break;
        default:
          PADDLE_THROW("Tensors with rank at most 6 are supported");
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
