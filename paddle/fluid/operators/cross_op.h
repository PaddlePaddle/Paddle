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
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;
const int kDefaultDim = framework::DDim::kMaxRank;

inline bool CheckDims(const DDim& dims_x, const DDim& dims_y) {
  if (dims_x.size() != dims_y.size()) {
    return false;
  }
  for (int i = 0; i < dims_x.size(); i++) {
    if (dims_x[i] != dims_y[i]) {
      return false;
    }
  }
  return true;
}

template <typename DeviceContext, typename T>
class CrossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input_x_var = context.InputVar("X");
    auto* input_y_var = context.InputVar("Y");
    auto* output_var = context.OutputVar("Out");

    auto& input_x = input_x_var->Get<LoDTensor>();
    auto& input_y = input_y_var->Get<LoDTensor>();
    auto* output = output_var->GetMutable<LoDTensor>();
    int dim = context.Attr<int>("dim");

    auto input_x_dims = input_x.dims();
    auto input_y_dims = input_y.dims();
    bool dims_match = CheckDims(input_x_dims, input_y_dims);
    PADDLE_ENFORCE_EQ(dims_match, true,
                      platform::errors::InvalidArgument(
                          "The 'shape' of Input(X) should be equal to "
                          "the 'shape' of Input(Y). But received "
                          "Input(X).dimensions = [%s], "
                          "Input(Y).dimensions = [%s]",
                          input_x_dims, input_x_dims));

    if (dim != kDefaultDim) {
      PADDLE_ENFORCE_EQ(
          dim < input_x_dims.size() && dim >= (0 - input_x_dims.size()), true,
          platform::errors::OutOfRange(
              "Attr(dim) is out of range, It's expected "
              "to be in range of [-%d, %d]. But received Attr(dim) = %d.",
              input_x_dims.size(), input_x_dims.size() - 1, dim));
      if (dim < 0) {
        dim += input_x_dims.size();
      }

      PADDLE_ENFORCE_EQ(
          input_x_dims[dim] == 3, true,
          platform::errors::InvalidArgument(
              "Input(X/Y).dims[dim] must be equal to 3. But received: "
              "Input(X/Y).dims[dim] = [%d].",
              input_x_dims[dim]));
    } else {
      for (auto i = 0; i < input_x_dims.size(); i++) {
        if (input_x_dims[i] == 3) {
          dim = i;
          break;
        }
      }
      PADDLE_ENFORCE_EQ(dim == kDefaultDim, false,
                        platform::errors::InvalidArgument(
                            "There must be at least one dimension 'd' so that "
                            "Input(X/Y).dims()[d] is equal to 3. "
                            "But received: Input(X/Y).dims() == [%s].",
                            input_x_dims));
    }
    auto outer_loops = 1;
    for (auto i = 0; i < dim; i++) {
      outer_loops *= input_x_dims[i];
    }
    auto slice_size = 1;
    for (auto i = dim + 1; i < input_x_dims.size(); i++) {
      slice_size *= input_x_dims[i];
    }

    std::vector<T> input_x_vec, input_y_vec;
    framework::TensorToVector(input_x, context.device_context(), &input_x_vec);
    framework::TensorToVector(input_y, context.device_context(), &input_y_vec);
    std::vector<T> out_vec(output->numel());

    output->mutable_data<T>(context.GetPlace());

    for (auto i = 0; i < outer_loops; i++) {
      for (auto j = 0; j < 3; j++) {
        auto dst_pos = (3 * i + j) * slice_size;
        auto in_pos1 = (3 * i + ((j + 1) % 3)) * slice_size;
        auto in_pos2 = (3 * i + ((j + 2) % 3)) * slice_size;

        for (auto k = 0; k < slice_size; k++) {
          out_vec[dst_pos + k] =
              input_x_vec[in_pos1 + k] * input_y_vec[in_pos2 + k] -
              input_x_vec[in_pos2 + k] * input_y_vec[in_pos1 + k];
        }
      }
    }
    framework::TensorFromVector(out_vec, context.device_context(), output);
    output->Resize(input_x_dims);
  }
};

template <typename DeviceContext, typename T>
class CrossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input_x_var = context.InputVar("X");
    auto* input_y_var = context.InputVar("Y");
    auto* input_out_grad_var = context.InputVar(framework::GradVarName("Out"));
    auto* output_x_grad_var = context.OutputVar(framework::GradVarName("X"));
    auto* output_y_grad_var = context.OutputVar(framework::GradVarName("Y"));

    auto& input_x = input_x_var->Get<LoDTensor>();
    auto& input_y = input_y_var->Get<LoDTensor>();
    auto& input_out_grad = input_out_grad_var->Get<LoDTensor>();
    auto* output_x_grad = output_x_grad_var->GetMutable<LoDTensor>();
    auto* output_y_grad = output_y_grad_var->GetMutable<LoDTensor>();

    int dim = context.Attr<int>("dim");
    auto input_x_dims = input_x.dims();
    if (dim != kDefaultDim) {
      PADDLE_ENFORCE_EQ(
          dim < input_x_dims.size() && dim >= (0 - input_x_dims.size()), true,
          platform::errors::OutOfRange(
              "Attr(dim) is out of range, It's expected "
              "to be in range of [-%d, %d]. But received Attr(dim) = %d.",
              input_x_dims.size(), input_x_dims.size() - 1, dim));
      if (dim < 0) {
        dim += input_x_dims.size();
      }

      PADDLE_ENFORCE_EQ(
          input_x_dims[dim] == 3, true,
          platform::errors::InvalidArgument(
              "Input(X/Y).dims[dim] must be equal to 3. But received: "
              "Input(X/Y).dims[dim] = [%d].",
              input_x_dims[dim]));
    } else {
      for (auto i = 0; i < input_x_dims.size(); i++) {
        if (input_x_dims[i] == 3) {
          dim = i;
          break;
        }
      }
      PADDLE_ENFORCE_EQ(dim == kDefaultDim, false,
                        platform::errors::InvalidArgument(
                            "There must be at least one dimension 'd' "
                            "so that Input(X/Y).dims()[d] is equal to 3. "
                            "But received: Input(X/Y).dims() == [%s].",
                            input_x_dims));
    }
    auto outer_loops = 1;
    for (auto i = 0; i < dim; i++) {
      outer_loops *= input_x_dims[i];
    }
    auto slice_size = 1;
    for (auto i = dim + 1; i < input_x_dims.size(); i++) {
      slice_size *= input_x_dims[i];
    }

    std::vector<T> input_x_vec, input_y_vec, input_dout_vec;
    framework::TensorToVector(input_x, context.device_context(), &input_x_vec);
    framework::TensorToVector(input_y, context.device_context(), &input_y_vec);
    framework::TensorToVector(input_out_grad, context.device_context(),
                              &input_dout_vec);
    std::vector<T> out_dx_vec(output_x_grad->numel());
    std::vector<T> out_dy_vec(output_y_grad->numel());

    output_x_grad->mutable_data<T>(context.GetPlace());
    output_y_grad->mutable_data<T>(context.GetPlace());

    for (auto i = 0; i < outer_loops; i++) {
      for (auto j = 0; j < 3; j++) {
        auto dst_pos = (3 * i + j) * slice_size;
        auto in_pos1 = (3 * i + ((j + 1) % 3)) * slice_size;
        auto in_pos2 = (3 * i + ((j + 2) % 3)) * slice_size;
        for (auto k = 0; k < slice_size; k++) {
          out_dx_vec[dst_pos + k] =
              input_dout_vec[in_pos2 + k] * input_y_vec[in_pos1 + k] -
              input_dout_vec[in_pos1 + k] * input_y_vec[in_pos2 + k];
          out_dy_vec[dst_pos + k] =
              input_dout_vec[in_pos1 + k] * input_x_vec[in_pos2 + k] -
              input_dout_vec[in_pos2 + k] * input_x_vec[in_pos1 + k];
        }
      }
    }
    framework::TensorFromVector(out_dx_vec, context.device_context(),
                                output_x_grad);
    framework::TensorFromVector(out_dy_vec, context.device_context(),
                                output_y_grad);
    output_x_grad->Resize(input_x_dims);
    output_y_grad->Resize(input_x_dims);
  }
};

}  // namespace operators
}  // namespace paddle
