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
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
template <typename Context, typename T, typename IndexT = int>
void IndexAddInner(const Context& ctx,
                   DenseTensor* input,
                   const DenseTensor& index,
                   int axis,
                   DenseTensor* add_value,
                   DenseTensor* output) {
  auto input_dim = input->dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = output->dims();
  auto index_size = index.dims()[0];
  auto add_value_dim = add_value->dims();

  PADDLE_ENFORCE_EQ(
      input_dim_size,
      add_value_dim.size(),
      phi::errors::InvalidArgument(
          "input_dim size expected == value_dim size, "
          "but got input_dim size is %ld and value_dim size is %ld. "
          "Please check inputs. ",
          input_dim_size,
          add_value_dim.size()));

  const IndexT* index_data = index.data<IndexT>();

  ctx.template Alloc<T>(output);

  // copy x to output.
  // todo(@limin29): inplace do not need copy.
  phi::Copy(ctx, *input, ctx.GetPlace(), false, output);

  auto slice_size = 1;
  for (auto i = axis + 1; i < input_dim_size; i++) {
    slice_size *= input_dim[i];
  }
  auto outer_nums = 1;
  for (auto i = 0; i < axis; i++) {
    outer_nums *= input_dim[i];
  }

  for (int i = 0; i < index_size; i++) {
    PADDLE_ENFORCE_GE(
        index_data[i],
        0,
        phi::errors::InvalidArgument(
            "Variable value (index) of OP(index_add) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[axis],
            index_data[i]));
    PADDLE_ENFORCE_LT(
        index_data[i],
        input_dim[axis],
        phi::errors::InvalidArgument(
            "Variable value (index) of OP(index_add) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[axis],
            index_data[i]));
  }

  VLOG(3) << "Index_Add_Debug; outer_nums: " << outer_nums
          << "; slice_size: " << slice_size << "; index_size: " << index_size;

  output->Resize(phi::make_ddim({outer_nums, input_dim[axis], slice_size}));
  add_value->Resize(phi::make_ddim({outer_nums, index_size, slice_size}));
  VLOG(3) << "output.dims: " << output->dims()
          << ", add_value.dims: " << add_value->dims();

  auto add_value_tensor = EigenTensor<T, 3>::From(*add_value);
  auto output_tensor = EigenTensor<T, 3>::From(*output);

  auto& place = *ctx.eigen_device();
  for (auto j = 0; j < index_size; j++) {
    IndexT index_value = index_data[j];
    auto output_t = output_tensor.chip(index_value, 1);
    output_t.device(place) = output_t + add_value_tensor.chip(j, 1);
  }
  output->Resize(output_dim);
  add_value->Resize(add_value_dim);
}

template <typename T, typename Context>
void IndexAddBaseKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& index,
                        int axis,
                        const DenseTensor& add_value,
                        DenseTensor* output) {
  const auto& index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT64 || index_type == phi::DataType::INT32;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  PADDLE_ENFORCE_EQ(
      axis < x.dims().size() && axis > -(x.dims().size()),
      true,
      phi::errors::InvalidArgument("axis expected < %ld, and > %ld, "
                                   "but got %ld. Please check the axis input. ",
                                   x.dims().size(),
                                   -(x.dims().size()),
                                   axis));

  if (axis < 0) {
    axis += x.dims().size();
  }

  for (int i = 0; i < x.dims().size(); i++) {
    if (i != axis) {
      PADDLE_ENFORCE_EQ(
          x.dims()[i],
          add_value.dims()[i],
          phi::errors::InvalidArgument(
              "The value.dims[i] should be equal to x.dims[i] when i != axis. "
              "but got value.dims[%d] = %ld, x.dims[%d] = %ld",
              i,
              add_value.dims()[i],
              i,
              x.dims()[i]));
    }
  }

  auto inputs = x;
  auto add_values = add_value;
  if (index_type == phi::DataType::INT32) {
    IndexAddInner<Context, T, int>(
        dev_ctx, &inputs, index, axis, &add_values, output);
  } else if (index_type == phi::DataType::INT64) {
    IndexAddInner<Context, T, int64_t>(
        dev_ctx, &inputs, index, axis, &add_values, output);
  }
}

}  // namespace phi
