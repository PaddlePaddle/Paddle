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
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using DDim = framework::DDim;

template <typename DeviceContext, typename T, typename IndexT = int>
void IndexSelectInner(const framework::ExecutionContext& context,
                      phi::DenseTensor* input,
                      const phi::DenseTensor& index,
                      phi::DenseTensor* output,
                      int dim) {
  auto input_dim = input->dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = output->dims();
  auto index_size = index.dims()[0];

  phi::DenseTensor index_cpu_copy;
  if (!platform::is_cpu_place(index.place())) {
    framework::TensorCopySync(index, platform::CPUPlace(), &index_cpu_copy);
  }
  const IndexT* index_data = platform::is_cpu_place(index.place())
                                 ? index.data<IndexT>()
                                 : index_cpu_copy.data<IndexT>();
  output->mutable_data<T>(context.GetPlace());

  auto slice_size = 1;
  for (auto i = dim + 1; i < input_dim_size; i++) {
    slice_size *= input_dim[i];
  }

  auto outer_nums = 1;
  for (auto i = 0; i < dim; i++) {
    outer_nums *= input_dim[i];
  }

  for (int i = 0; i < index_size; i++) {
    PADDLE_ENFORCE_GE(
        index_data[i],
        0,
        platform::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[dim],
            index_data[i]));
    PADDLE_ENFORCE_LT(
        index_data[i],
        input_dim[dim],
        platform::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[dim],
            index_data[i]));
  }

  VLOG(3) << "Index_Select_Debug; outer_nums: " << outer_nums
          << "; slice_size: " << slice_size << "; index_size: " << index_size;

  input->Resize(phi::make_ddim({outer_nums, input_dim[dim], slice_size}));
  output->Resize(phi::make_ddim({outer_nums, index_size, slice_size}));

  auto input_tensor = phi::EigenTensor<T, 3>::From(*input);
  auto output_tensor = phi::EigenTensor<T, 3>::From(*output);

  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();

  for (auto j = 0; j < index_size; j++) {
    IndexT index_value = index_data[j];
    auto output_t = output_tensor.chip(j, 1);
    output_t.device(place) = input_tensor.chip(index_value, 1);
  }
  input->Resize(input_dim);
  output->Resize(output_dim);
}

template <typename DeviceContext, typename T, class Enable = void>
struct IndexSelectAdd {
  void operator()(const framework::ExecutionContext& ctx,
                  int slice_size,
                  const T* src_pointer,
                  const T* p_pointer,
                  T* dist_pointer) {
    for (int i = 0; i < slice_size; i++) {
      dist_pointer[i] = src_pointer[i] + p_pointer[i];
    }
  }
};
template <typename DeviceContext, typename T>
struct IndexSelectAdd<
    DeviceContext,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext& ctx,
                  int slice_size,
                  const T* src_pointer,
                  const T* p_pointer,
                  T* dist_pointer) {
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(ctx);
    blas.VADD(slice_size, src_pointer, p_pointer, dist_pointer);
  }
};

template <typename DeviceContext, typename T, typename IndexT = int>
void IndexSelectGradInner(const framework::ExecutionContext& context,
                          const phi::DenseTensor& out_grad,
                          const phi::DenseTensor& index,
                          phi::DenseTensor* x_grad,
                          int dim) {
  const T* input_data = out_grad.data<T>();
  const IndexT* index_data = index.data<IndexT>();
  const T* p_output = x_grad->mutable_data<T>(context.GetPlace());
  T* out_data = x_grad->mutable_data<T>(context.GetPlace());
  auto input_dim = out_grad.dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = x_grad->dims();

  auto& dev_ctx = context.template device_context<DeviceContext>();
  phi::funcs::SetConstant<DeviceContext, T> set_constant;
  set_constant(dev_ctx, x_grad, static_cast<T>(0.0));

  auto slice_size = 1;
  for (auto i = dim + 1; i < input_dim_size; i++) {
    slice_size *= input_dim[i];
  }

  auto input_width = slice_size * input_dim[dim];
  auto output_width = slice_size * output_dim[dim];

  auto outer_nums = 1;
  for (auto i = 0; i < dim; i++) {
    outer_nums *= input_dim[i];
  }

  auto index_size = index.dims()[0];
  VLOG(3) << "Index_Select_Grad_Debug; outer_nums: " << outer_nums
          << "; slice_size: " << slice_size << "; input_width: " << input_width
          << "; output_width: " << output_width
          << "; index_size: " << index_size;

  for (auto i = 0; i < outer_nums; i++) {
    auto input_start_offset = i * input_width;
    auto output_start_offset = i * output_width;

    for (auto j = 0; j < index_size; j++) {
      IndexT index_value = index_data[j];
      auto src = input_data + input_start_offset + j * slice_size;
      auto p_out = p_output + output_start_offset + index_value * slice_size;
      auto dst = out_data + output_start_offset + index_value * slice_size;
      IndexSelectAdd<DeviceContext, T> index_select_add;
      index_select_add(context, slice_size, src, p_out, dst);
    }
  }
  x_grad->Resize(output_dim);
}

}  // namespace operators
}  // namespace paddle
