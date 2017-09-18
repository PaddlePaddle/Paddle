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

template <typename Place, typename T, int Dims>
void EigenTranspose(const framework::ExecutionContext& context,
                    const framework::Tensor& in, framework::Tensor& out,
                    std::vector<int> axis) {
  Eigen::array<int, Dims> permute;
  for (int i = 0; i < Dims; i++) {
    permute[i] = axis[i];
  }
  auto in_dim = in.dims();
  auto out_dim = out.dims();

  auto eigen_in = framework::EigenTensor<T, Dims>::From(in);
  auto eigen_out = framework::EigenTensor<T, Dims>::From(out);
  auto& dev = context.GetEigenDevice<Place>();
  eigen_out.device(dev) = eigen_in.shuffle(permute);
}

template <typename Place, typename T>
class TransposeKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto* output = context.Output<framework::Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    auto axis = context.Attr<std::vector<int>>("axis");
    int ndims = axis.size();
    switch (ndims) {
      case 1:
        break;
      case 2:
        EigenTranspose<Place, T, 2>(context, *input, *output, axis);
        break;
      case 3:
        EigenTranspose<Place, T, 3>(context, *input, *output, axis);
        break;
      case 4:
        EigenTranspose<Place, T, 4>(context, *input, *output, axis);
        break;
      case 5:
        EigenTranspose<Place, T, 5>(context, *input, *output, axis);
        break;
      case 6:
        EigenTranspose<Place, T, 6>(context, *input, *output, axis);
        break;
      default:
        PADDLE_THROW("Tensors with rank at most 6 are supported");
    }
  }
};

template <typename Place, typename T>
class TransposeGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* output_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Output"));
    auto* input_grad =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    input_grad->mutable_data<T>(context.GetPlace());

    auto axis_temp = context.Attr<std::vector<int>>("axis");
    std::vector<int> axis(axis_temp);

    for (size_t i = 0; i < axis.size(); i++) {
      axis[axis_temp[i]] = i;
    }

    int ndims = axis.size();

    switch (ndims) {
      case 1:
        break;
      case 2:
        EigenTranspose<Place, T, 2>(context, *output_grad, *input_grad, axis);
        break;
      case 3:
        EigenTranspose<Place, T, 3>(context, *output_grad, *input_grad, axis);
        break;
      case 4:
        EigenTranspose<Place, T, 4>(context, *output_grad, *input_grad, axis);
        break;
      case 5:
        EigenTranspose<Place, T, 5>(context, *output_grad, *input_grad, axis);
        break;
      case 6:
        EigenTranspose<Place, T, 6>(context, *output_grad, *input_grad, axis);
        break;
      default:
        PADDLE_THROW("Tensors with rank at most 6 are supported");
    }
  }
};

}  // namespace operators
}  // namespace paddle
