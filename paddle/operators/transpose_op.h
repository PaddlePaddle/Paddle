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

#include <iostream>
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
void NaiveCpuTranspose(const framework::ExecutionContext& context,
                       const framework::Tensor& in, framework::Tensor& out,
                       std::vector<int> axis) {
  auto in_data = in.data<T>();
  auto out_data = out.mutable_data<T>(context.GetPlace());
  auto in_dim = in.dims();
  auto out_dim = out.dims();
  size_t ndims = in_dim.size();

  std::vector<int> in_offset(ndims, 1);
  std::vector<int> out_offset(ndims, 1);

  for (int i = ndims - 2; i >= 0; i--) {
    in_offset[i] = in_offset[i + 1] * in_dim[i + 1];
    out_offset[i] = out_offset[i + 1] * out_dim[i + 1];
  }

  size_t data_size = product(in_dim);

  for (size_t to_index = 0; to_index < data_size; to_index++) {
    int from_index = 0;
    int temp = to_index;
    for (size_t i = 0; i < ndims; i++) {
      from_index += (temp / out_offset[i]) * in_offset[axis[i]];
      temp = temp % out_offset[i];
    }
    out_data[to_index] = in_data[from_index];
  }
}

template <typename Place, typename T, int Dims>
void DoTranspose(const framework::ExecutionContext& context,
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
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    auto axis = context.GetAttr<std::vector<int>>("axis");
    int ndims = axis.size();
    switch (ndims) {
      case 2:
        DoTranspose<Place, T, 2>(context, *in, *out, axis);
        break;
      case 3:
        DoTranspose<Place, T, 3>(context, *in, *out, axis);
        break;
      case 4:
        DoTranspose<Place, T, 4>(context, *in, *out, axis);
        break;
      case 5:
        DoTranspose<Place, T, 5>(context, *in, *out, axis);
        break;
      default:
        NaiveCpuTranspose<Place, T>(context, *in, *out, axis);
        break;
    }
  }
};

template <typename Place, typename T>
class TransposeGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* out = context.Output<framework::Tensor>(framework::GradVarName("X"));
    out->mutable_data<T>(context.GetPlace());

    auto axis_temp = context.GetAttr<std::vector<int>>("axis");
    std::vector<int> axis(axis_temp);

    for (size_t i = 0; i < axis.size(); i++) {
      axis[axis_temp[i]] = i;
    }

    int ndims = axis.size();

    switch (ndims) {
      case 2:
        DoTranspose<Place, T, 2>(context, *in, *out, axis);
        break;
      case 3:
        DoTranspose<Place, T, 3>(context, *in, *out, axis);
        break;
      case 4:
        DoTranspose<Place, T, 4>(context, *in, *out, axis);
        break;
      case 5:
        DoTranspose<Place, T, 5>(context, *in, *out, axis);
        break;
      default:
        NaiveCpuTranspose<Place, T>(context, *in, *out, axis);
        break;
    }
  }
};

}  // namespace operators
}  // namespace paddle
