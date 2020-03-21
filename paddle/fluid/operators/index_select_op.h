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

#ifndef PADDLE_FLUID_OPERATORS_INDEX_SELECT_OP_H_
#define PADDLE_FLUID_OPERATORS_INDEX_SELECT_OP_H_

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

template <typename T, typename IndexT = int>
IndexSelectInner(const framework::ExecutionContext& context,
                 const Tensor& input, const Tensor& index, Tensor* output,
                 int dim) {
  auto index_size = index.dims()[0];

  const T* p_input = input.data<T>(context.GetPlace());
  const IndexT* p_index = index.data<IndexT>(context.GetPlace());
  T* p_output = output->data<T>(context.GetPlace());

  auto input_dim = input.dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = output->dims();

  auto slice_size = 1;
  for (size_t i = dim + 1; i < input_dim_size; i++) {
    slice_size *= input_dim[i];
  }
  const size_t slice_bytes = slice_size * sizeof(T);

  auto input_width = slice_size * input_dim[dim];
  auto output_width = slice_size * output_dim[dim];

  auto outer_nums = 1;
  for (auto i = 0; i < dim; i++) {
    outer_nums *= input_dim[i];
  }

  VLOG(1) << "outer_nums: " << outer_nums << "; slice_size: " << slice_size
          << "; input_width: " << input_width
          << "; output_width: " << output_width;

  for (auto i = 0; i < outer_nums; i++) {
    auto input_start_offset = i * input_width;
    auto output_start_offset = i * output_width;

    for (auto j = 0; j < index_size; j++) {
      IndexT index_value = p_index[j];
      memcpy(p_output + output_start_offset + j * slice_size,
             p_input + input_start_offset + index_value * slice_size,
             slice_bytes);
    }
  }
}

template <typename T>
class IndexSelectKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) {
    auto* inputs = context.Input<LoDTensor>("X");
    auto* index = context.Input<LoDTensor>("Index");
    auto* output = context.Output<LoDTensor>("Out");
    auto dim = context.Attr<int>("dim");

    if (index.dims().size() == 2) {
      PADDLE_ENFORCE_EQ(
          index.dims()[1], 1,
          "index.dims()[1] should be 1 when index.dims().size() == "
          "2 in index_select_op.");
    } else {
      PADDLE_ENFORCE_EQ(
          index.dims().size(), 1,
          "index.dims().size() should be 1 or 2 in index_select_op.");
    }

    const auto& index_type = index->type();
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE(
        index_type_match,
        "Index holds the wrong type, it holds %s, but desires to be %s or %s",
        paddle::framework::DataTypeToString(index_type),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT32),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT64));

    if (index_type == framework::proto::VarType::INT32) {
      IndexSelectInner<T, int>(ctx.device_context(), *inputs, *index, output,
                               dim);
    } else if (index_type == framework::proto::VarType::INT64) {
      IndexSelectInner<T, int64_t>(ctx.device_context(), *inputs, *index,
                                   output, dim);
    }
  }
}

template <typename T, typename IndexT = int>
IndexSelectGradInner(const framework::ExecutionContext& context,
                     const Tensor& out_grad, const Tensor& index,
                     Tensor* x_grad, int dim) {
  auto index_size = index.dims()[0];

  const T* p_input = out_grad.data<T>(context.GetPlace());
  const IndexT* p_index = index.data<IndexT>(context.GetPlace());
  T* p_output = x_grad->data<T>(context.GetPlace());

  auto input_dim = input.dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = output->dims();

  auto slice_size = 1;
  for (size_t i = dim + 1; i < input_dim_size; i++) {
    slice_size *= input_dim[i];
  }
  const size_t slice_bytes = slice_size * sizeof(T);

  auto input_width = slice_size * input_dim[dim];
  auto output_width = slice_size * output_dim[dim];

  auto outer_nums = 1;
  for (auto i = 0; i < dim; i++) {
    outer_nums *= input_dim[i];
  }

  VLOG(1) << "index_select_grad: outer_nums: " << outer_nums
          << "; slice_size: " << slice_size << "; input_width: " << input_width
          << "; output_width: " << output_width;

  for (auto i = 0; i < outer_nums; i++) {
    auto input_start_offset = i * input_width;
    auto output_start_offset = i * output_width;

    for (auto j = 0; j < index_size; ++j) {
      IndexT index_value = p_index[i];
      memset(p_output + output_start_offset + slice_size * index_value, 0,
             slice_bytes);
    }
    for (auto j = 0; j < index_size; j++) {
      IndexT index_value = p_index[j];
      T* ou = p_output + output_start_offset + index_value * slice_size;
      T* in = p_input + input_start_offset + j * slice_size;
      for (auto k = 0; k < slice_size; k++) {
        *(ou + k) += *(in + k);
      }
    }
  }
}

template <typename T>
class IndexSelectGradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* index = ctx.Input<LodTensor>("Index");
    auto* x = ctr.Input<LodTensor>("X");
    auto* x_grad = ctx.Output<LodTensor>(framework::GradVarName("X"));
    auto* out_grad = ctx.Input<LodTensor>(framework::GradVarName("Out"));

    const auto& index_type = index->type();
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE(
        index_type_match,
        "Index holds the wrong type, it holds %s, but desires to be %s or %s",
        paddle::framework::DataTypeToString(index_type),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT32),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT64));
    if (index_type == framework::proto::VarType::INT32) {
      IndexSelectGradInner<T, int>(ctx.device_context(), *out_grad, *index,
                                   x_grad, dim);
    } else if (index_type == framework::proto::VarType::INT64) {
      IndexSelectGradInner<T, int64_t>(ctx.device_context(), *out_grad, *index,
                                       x_grad, dim);
    }
  }
}

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_FLUID_OPERATORS_INDEX_SELECT_OP_H_
