// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reduce_ops/reduce_op.h"

namespace paddle {
namespace operators {

// use for loop to speed up Eigen broadcast. 4 timer faster then broadcast
template <typename DeviceContext, typename T, typename Functor,
          bool kNoNeedBufferX = false>
class ReduceSumGradKernel : public framework::OpKernel<T> {
 public:
  void ComputeFromInput(const Tensor* input2,
                        const framework::ExecutionContext& context) const {
    auto dims = context.Attr<std::vector<int>>("dim");
    auto* input0 = context.Input<Tensor>("X");

    auto* output = context.Output<Tensor>(framework::GradVarName("X"));
    output->mutable_data<T>(context.GetPlace());
    const auto* input2_d = input2->data<T>();
    auto* output_d = output->data<T>();

    // handle reduce_all
    if (input2->dims().size() == 1 && input2->dims()[0] == 1) {
      for (int64_t i = 0; i < phi::product(input0->dims()); ++i) {
        output_d[i] = input2_d[0];
      }
      return;
    }

    // handle reduce by one dimension
    int reduce_dim_index = dims[0];
    if (reduce_dim_index < 0) {
      reduce_dim_index += input0->dims().size();
    }

    auto& input_dim = input0->dims();
    int64_t before_dim = 1;
    for (int i = 0; i < reduce_dim_index; ++i) {
      before_dim *= input_dim[i];
    }
    int64_t reduce_dim = input_dim[reduce_dim_index];
    int64_t after_dim = 1;
    for (int i = reduce_dim_index + 1; i < input_dim.size(); ++i) {
      after_dim *= input_dim[i];
    }
    for (int64_t i = 0; i < before_dim; ++i) {
      for (int64_t j = 0; j < reduce_dim; ++j) {
        for (int64_t k = 0; k < after_dim; ++k) {
          output_d[i * reduce_dim * after_dim + j * after_dim + k] =
              input2_d[i * after_dim + k];
        }
      }
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto dims = context.Attr<std::vector<int>>("dim");
    if (context.GetPlace().GetType() == platform::CPUPlace().GetType() &&
        dims.size() == 1) {
      int in_dtype = context.Attr<int>("out_dtype");

      if (in_dtype >= 0) {
        Tensor tmp_tensor;
        auto* pre_input = context.Input<Tensor>(framework::GradVarName("Out"));
        auto in_kernel_type = framework::OpKernelType(
            framework::TransToProtoVarType(pre_input->dtype()),
            context.GetPlace());
        auto out_kernel_type = framework::OpKernelType(
            static_cast<framework::proto::VarType::Type>(in_dtype),
            context.GetPlace());
        framework::TransDataType(in_kernel_type, out_kernel_type, *pre_input,
                                 &tmp_tensor);
        ComputeFromInput(&tmp_tensor, context);
      } else {
        auto* input2 = context.Input<Tensor>(framework::GradVarName("Out"));
        ComputeFromInput(input2, context);
      }
      return;
    }
    // default use Eigen broadcast
    ReduceGradKernel<DeviceContext, T, Functor, kNoNeedBufferX> kernel;
    kernel.Compute(context);
  }
};

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

}  // namespace operators
}  // namespace paddle
