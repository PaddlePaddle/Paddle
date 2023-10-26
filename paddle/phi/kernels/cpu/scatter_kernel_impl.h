// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/cpu/index_add_impl.h"

#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/index_add_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename Context, typename T, typename IndexT = int>
void IndexReduceInner(const Context& ctx,
                      DenseTensor* input,
                      const DenseTensor& index,
                      int axis,
                      const std::string& reduce,
                      bool include_self,
                      DenseTensor* source,
                      DenseTensor* output) {
  auto input_dim = input->dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = output->dims();
  auto index_size = index.dims()[0];
  auto source_dim = source->dims();

  const IndexT* index_data = index.data<IndexT>();
  ctx.template Alloc<T>(output);

  auto zeros = Full<T, Context>(ctx, vectorize(input_dim), 0);
  auto ones = Full<T, Context>(ctx, vectorize(input_dim), 1);
  auto counts = include_self ? ones : zeros;
  auto src_ones = Full<T, Context>(ctx, vectorize(source->dims()), 1);
  auto src_cnts = IndexAdd<T, Context>(ctx, counts, index, src_ones, axis);
  auto mask = Equal<T, Context>(ctx, src_cnts, zeros);

  if (include_self) {
    phi::Copy(ctx, *input, ctx.GetPlace(), false, output);
  } else {
    T init_val;
    if (reduce == "mul" || reduce == "multiply") {
      init_val = static_cast<T>(1);
    } else if (reduce == "amin") {
      init_val = std::numeric_limits<T>::has_infinity
                     ? std::numeric_limits<T>::infinity()
                     : std::numeric_limits<T>::max();
    } else if (reduce == "amax") {
      init_val = std::numeric_limits<T>::has_infinity
                     ? -std::numeric_limits<T>::infinity()
                     : std::numeric_limits<T>::lowest();
    } else {
      init_val = static_cast<T>(0);
    }
    auto init = Full<T, Context>(ctx, vectorize(input_dim), init_val);

    auto out = Where<T, Context>(ctx, mask, *input, init);
    phi::Copy(ctx, out, ctx.GetPlace(), false, output);
  }

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

  output->Resize(phi::make_ddim({outer_nums, input_dim[axis], slice_size}));
  source->Resize(phi::make_ddim({outer_nums, index_size, slice_size}));

  auto source_tensor = EigenTensor<T, 3>::From(*source);
  auto output_tensor = EigenTensor<T, 3>::From(*output);

  auto& place = *ctx.eigen_device();
  for (auto j = 0; j < index_size; j++) {
    IndexT index_value = index_data[j];
    auto output_t = output_tensor.chip(index_value, 1);
    auto source_t = source_tensor.chip(j, 1);
    if (reduce == "add" || reduce == "mean") {
      output_t.device(place) = output_t + source_t;
    } else if (reduce == "mul" || reduce == "muliply") {
      output_t.device(place) = output_t * source_t;
    } else if (reduce == "amin") {
      output_t.device(place) = output_t.cwiseMin(source_t);
    } else if (reduce == "amax") {
      output_t.device(place) = output_t.cwiseMax(source_t);
    } else if (reduce == "assign") {
      output_t.device(place) = source_t;
    }
  }

  output->Resize(output_dim);
  source->Resize(source_dim);

  if (reduce == "mean") {
    auto src_cnts_wo_zeros = Where<T, Context>(ctx, mask, ones, src_cnts);
    auto out = Divide<T, Context>(ctx, *output, src_cnts_wo_zeros);
    phi::Copy(ctx, out, ctx.GetPlace(), false, output);
  }
}

template <typename T, typename Context>
void IndexReduceBaseKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& index,
                           const DenseTensor& source,
                           int axis,
                           const std::string& reduce,
                           bool include_self,
                           DenseTensor* output) {
  const auto& index_type = index.dtype();

  if (axis < 0) {
    axis += x.dims().size();
  }

  PADDLE_ENFORCE_LT(
      axis,
      x.dims().size(),
      phi::errors::InvalidArgument(
          "Axis value (axis) of OP(scatter) "
          "expected >= 0 and < %ld, but got %ld. Please check axis "
          "value.",
          x.dims().size(),
          axis));

  auto inputs = x;
  auto src = source;
  if (index_type == phi::DataType::INT32) {
    IndexReduceInner<Context, T, int32_t>(
        dev_ctx, &inputs, index, axis, reduce, include_self, &src, output);
  } else if (index_type == phi::DataType::INT64) {
    IndexReduceInner<Context, T, int64_t>(
        dev_ctx, &inputs, index, axis, reduce, include_self, &src, output);
  }
}

}  // namespace phi
