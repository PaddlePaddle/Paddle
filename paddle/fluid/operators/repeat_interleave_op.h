// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

#include "paddle/fluid/operators/index_select_op.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename RepeatsT = int>
void RepeatsTensor2IndexTensor(const LoDTensor& repeats, LoDTensor* index) {
  LoDTensor repeats_cpu_copy;
  if (!platform::is_cpu_place(repeats.place())) {
    framework::TensorCopySync(repeats, platform::CPUPlace(), &repeats_cpu_copy);
  }
  const RepeatsT* repeats_data = platform::is_cpu_place(repeats.place())
                                     ? repeats.data<RepeatsT>()
                                     : repeats_cpu_copy.data<RepeatsT>();

  int64_t index_size = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    index_size += repeats_data[i];
  }
  std::vector<RepeatsT> index_vec(index_size);
  int offset = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    std::fill_n(index_vec.begin() + offset, repeats_data[i], i);
    offset += repeats_data[i];
  }
  index->Resize(framework::make_ddim({index_size}));

  auto ctx =
      paddle::platform::DeviceContextPool::Instance().Get(repeats.place());
  paddle::framework::TensorFromVector<RepeatsT>(index_vec, *ctx, index);
}

template <typename DeviceContext, typename T>
class RepeatInterleaveKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto inputs = *context.Input<framework::LoDTensor>("X");
    auto* output = context.Output<framework::LoDTensor>("Out");

    int dim = context.Attr<int>("dim");
    if (dim < 0) {
      dim += inputs.dims().size();
    }

    int repeats = context.Attr<int>("Repeats");
    framework::LoDTensor index;
    if (context.HasInput("RepeatsTensor")) {
      auto repeats_tensor =
          context.Input<framework::LoDTensor>("RepeatsTensor");

      PADDLE_ENFORCE_EQ(repeats_tensor->dims()[0] == inputs.dims()[dim], true,
                        platform::errors::InvalidArgument(
                            "The length of Input(RepeatsTensor) must be the "
                            "same as length of Input(X) in axis. "
                            "But received: [%s], required: [%d].",
                            repeats_tensor->dims()[0], inputs.dims()[dim]));

      const auto& index_type = repeats_tensor->type();
      bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                              index_type == framework::proto::VarType::INT64;
      PADDLE_ENFORCE_EQ(
          index_type_match, true,
          platform::errors::InvalidArgument(
              "Input(RepeatsTensor) holds the wrong type, it holds %s, but "
              "desires to be %s or %s",
              paddle::framework::DataTypeToString(index_type),
              paddle::framework::DataTypeToString(
                  framework::proto::VarType::INT32),
              paddle::framework::DataTypeToString(
                  framework::proto::VarType::INT64)));

      if (index_type == framework::proto::VarType::INT32) {
        RepeatsTensor2IndexTensor<DeviceContext, int>(*repeats_tensor, &index);
        auto output_dim = framework::vectorize(inputs.dims());
        output_dim[dim] = index.dims()[0];
        output->Resize(framework::make_ddim(output_dim));
        IndexSelectInner<DeviceContext, T, int>(context, &inputs, index, output,
                                                dim);
      } else if (index_type == framework::proto::VarType::INT64) {
        RepeatsTensor2IndexTensor<DeviceContext, int64_t>(*repeats_tensor,
                                                          &index);
        auto output_dim = framework::vectorize(inputs.dims());
        output_dim[dim] = index.dims()[0];
        output->Resize(framework::make_ddim(output_dim));
        IndexSelectInner<DeviceContext, T, int64_t>(context, &inputs, index,
                                                    output, dim);
      }
    } else if (repeats > 0) {
      int64_t index_size = inputs.dims()[dim] * repeats;
      std::vector<int> index_vec(index_size);
      for (int i = 0; i < inputs.dims()[dim]; i++) {
        std::fill_n(index_vec.begin() + i * repeats, repeats, i);
      }
      index.Resize(framework::make_ddim({index_size}));
      paddle::framework::TensorFromVector<int>(index_vec, &index);

      auto output_dim = framework::vectorize(inputs.dims());
      output_dim[dim] = index_size;
      output->Resize(framework::make_ddim(output_dim));

      IndexSelectInner<DeviceContext, T, int>(context, &inputs, index, output,
                                              dim);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "repeats must given with RepeatsTensor (tensor) or repeats (int)"));
    }
  }
};

template <typename DeviceContext, typename T>
class RepeatInterleaveGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x_grad =
        context.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto* out_grad =
        context.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    int dim = context.Attr<int>("dim");
    if (dim < 0) {
      dim += out_grad->dims().size();
    }

    int repeats = context.Attr<int>("Repeats");
    framework::LoDTensor index;
    if (context.HasInput("RepeatsTensor")) {
      auto repeats_tensor =
          context.Input<framework::LoDTensor>("RepeatsTensor");
      const auto& index_type = repeats_tensor->type();

      bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                              index_type == framework::proto::VarType::INT64;
      PADDLE_ENFORCE_EQ(
          index_type_match, true,
          platform::errors::InvalidArgument(
              "Input(Repeats) holds the wrong type, it holds %s, but "
              "desires to be %s or %s",
              paddle::framework::DataTypeToString(index_type),
              paddle::framework::DataTypeToString(
                  framework::proto::VarType::INT32),
              paddle::framework::DataTypeToString(
                  framework::proto::VarType::INT64)));

      if (index_type == framework::proto::VarType::INT32) {
        RepeatsTensor2IndexTensor<DeviceContext, int>(*repeats_tensor, &index);
        IndexSelectGradInner<DeviceContext, T, int>(context, *out_grad, index,
                                                    x_grad, dim);
      } else if (index_type == framework::proto::VarType::INT64) {
        RepeatsTensor2IndexTensor<DeviceContext, int64_t>(*repeats_tensor,
                                                          &index);
        IndexSelectGradInner<DeviceContext, T, int64_t>(context, *out_grad,
                                                        index, x_grad, dim);
      }
    } else if (repeats > 0) {
      int64_t index_size = x_grad->dims()[dim] * repeats;
      std::vector<int> index_vec(index_size);
      for (int i = 0; i < x_grad->dims()[dim]; i++) {
        std::fill_n(index_vec.begin() + i * repeats, repeats, i);
      }
      index.Resize(framework::make_ddim({index_size}));
      paddle::framework::TensorFromVector<int>(index_vec, &index);

      IndexSelectGradInner<DeviceContext, T, int>(context, *out_grad, index,
                                                  x_grad, dim);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "repeats must given with RepeatsTensor (tensor) or repeats (int)"));
    }
  }
};

}  // namespace operators
}  // namespace paddle
