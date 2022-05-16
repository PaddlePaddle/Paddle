// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
template <typename Context, typename T>
void IndexAddInner(const Context& ctx,
                    const DenseTensor& index,
                    const DenseTensor& x,
                    // if you want to Resize x, maybe 
                    // you should delete 'const'
                    // DenseTensor& x,
                    DenseTensor* output,
                    int axis,
                    T add_val,
                    DenseTensor* add_grad) {
  auto output_dim = x.dims();
  auto output_dim_size = output_dim.size();
  auto index_size = index.dims()[0];

  //  DenseTensor index_cpu_copy;
  //  if (!paddle::platform::is_cpu_place(index.place())) {
  //    phi::Copy(ctx, index, phi::CPUPlace(), true, &index_cpu_copy);
  //  }
  const int64_t* index_data = index.data<int64_t>();

  auto slice_size = 1;
  for (auto i = axis + 1; i < output_dim_size; i++) {
    slice_size *= output_dim[i];
  }

  auto outer_nums = 1;
  for (auto i = 0; i < axis; i++) {
    outer_nums *= output_dim[i];
  }

  for (int i = 0; i < index_size; i++) {
    bool check_index = index_data[i] >= 0 && index_data[i] < output_dim[axis];
    // PADDLE_ENFORCE_GE(
    PADDLE_ENFORCE_EQ(
        check_index,
        true,
        phi::errors::InvalidArgument(
            "Variable value (index) of OP(index_add) "
            "expected >= 0 and < %ld, but got %ld. Please check input or index"
            "value.",
            output_dim[axis],
            index_data[i]));
  }

  auto& place = *ctx.eigen_device();
  if (add_grad) {
    add_grad->Resize(phi::make_ddim({outer_nums, output_dim[axis], slice_size}));
    // if x is const, the line of code may be a bug
    // x.Resize(phi::make_ddim({outer_nums, output_dim[axis], slice_size}));

    auto x_tensor = EigenTensor<T, 3>::From(x);
    auto add_grad_tensor = EigenTensor<T, 3>::From(*add_grad);
    for (auto j = 0; j < index_size; j++) {
      int64_t index_value = index_data[j];
      auto add_grad_t = add_grad_tensor.chip(index_value, 1);
      add_grad_t.device(place) = x_tensor.chip(index_value, 1);
    }

    add_grad->Resize(output_dim);
    // x.Resize(output_dim);
  }

  if (output) {
    output->Resize(phi::make_ddim({outer_nums, output_dim[axis], slice_size}));
    auto output_tensor = EigenTensor<T, 3>::From(*output);
    for (auto j = 0; j < index_size; j++) {
      int64_t index_value = index_data[j];
      auto output_t = output_tensor.chip(index_value, 1);
      // output_t.device(place) = output_t.constant(add_val);
      output_t.device(place) += output_t.constant(add_val);
    }
    output->Resize(output_dim);
  }
}

template <typename T, typename Context>
void IndexAddBaseKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                        //  DenseTensor& x,
                         const IntArray& index_arr,
                         const Scalar& axis_scalar,
                         float add_value,
                         DenseTensor* output,
                         DenseTensor* add_grad) {
  auto index_list = index_arr.GetData();
  int64_t index_size = static_cast<int64_t>(index_list.size());
  // TODO:why create a DenseTensor for "index" and copy the data,
  // is this necessary ?
  DenseTensor index;
  index.Resize(make_ddim({index_size}));
  int64_t* index_ptr = dev_ctx.template Alloc<int64_t>(&index);
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       index_ptr,
                       dev_ctx.GetPlace(),
                       index_list.data(),
                       index_size * sizeof(int64_t));

  int axis = axis_scalar.to<int>();
  if (output) phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, output);
  if (axis < 0) {
    axis += x.dims().size();
  }
  IndexAddInner<Context, T>(
      dev_ctx, index, x, output, axis, static_cast<T>(add_value), add_grad);
}

}  // namespace phi