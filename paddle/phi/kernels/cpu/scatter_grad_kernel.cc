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

#include "paddle/phi/kernels/scatter_grad_kernel.h"
#include "glog/logging.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/bitwise_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/funcs/gather.h"
#include "paddle/phi/kernels/funcs/scatter.h"
#include "paddle/phi/kernels/index_add_kernel.h"
#include "paddle/phi/kernels/index_select_kernel.h"
#include "paddle/phi/kernels/put_along_axis_kernel.h"
#include "paddle/phi/kernels/reduce_any_kernel.h"
#include "paddle/phi/kernels/scatter_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename T, typename Context>
void ScatterGradKernel(const Context &ctx,
                       const DenseTensor &x,
                       const DenseTensor &index,
                       const DenseTensor &source,
                       const DenseTensor &out,
                       const DenseTensor &out_grad,
                       bool overwrite,
                       int axis,
                       const std::string &reduce,
                       bool include_self,
                       DenseTensor *x_grad,
                       DenseTensor *updates_grad) {
  const auto &index_type = index.dtype();
  PADDLE_ENFORCE_EQ(
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64,
      true,
      phi::errors::InvalidArgument(
          "scatter_op index holds the wrong type, it holds [%s],"
          "but desires to be [%s] or [%s]",
          index_type,
          phi::DataType::INT32,
          phi::DataType::INT64));

  PADDLE_ENFORCE_EQ(
      reduce == "add" || reduce == "mul" || reduce == "muliply" ||
          reduce == "mean" || reduce == "amin" || reduce == "amax",
      true,
      phi::errors::InvalidArgument(
          "Reduce holds the wrong value, it holds [%s],"
          "but desires to be add, mul, multiply, mean, amin, amax.",
          reduce));

  if (axis < 0) {
    axis += out_grad.dims().size();
  }

  std::string reducer = reduce;
  if (overwrite) {
    reducer = "assign";
  }

  DenseTensor new_index = index;
  DenseTensor new_source = source;
  if (index.dims().size() == 0) {
    new_index.Resize({1});

    if (source.dims().size() == x.dims().size() - 1) {
      auto dims = vectorize(source.dims());
      dims.insert(dims.begin(), 1);
      new_source.Resize(make_ddim(dims));
    }
  }

  if (x_grad) {
    ctx.template Alloc<T>(x_grad);
  }

  if (updates_grad) {
    ctx.template Alloc<T>(updates_grad);
  }

  if (reducer == "add") {
    if (x_grad) {
      if (include_self) {
        phi::Copy(ctx, out_grad, ctx.GetPlace(), false, x_grad);
      } else {
        *x_grad = Full<T, Context>(ctx, vectorize(out_grad.dims()), 0);
      }
    }

    if (updates_grad) {
      *updates_grad = IndexSelect<T, Context>(ctx, out_grad, index, axis);
    }
  } else if (reducer == "mean") {
    auto zeros = Full<T, Context>(ctx, vectorize(out_grad.dims()), 0);
    auto ones = Full<T, Context>(ctx, vectorize(out_grad.dims()), 1);
    auto counts = include_self ? ones : zeros;

    auto src_ones = Full<T, Context>(ctx, vectorize(new_source.dims()), 1);
    auto src_cnts =
        IndexAdd<T, Context>(ctx, counts, new_index, src_ones, axis);

    auto mask = Equal<T, Context>(ctx, src_cnts, zeros);

    auto N = Where<T, Context>(ctx, mask, ones, src_cnts);

    if (x_grad) {
      *x_grad = Divide<T, Context>(ctx, out_grad, N);
    }

    if (updates_grad) {
      auto N_src = IndexSelect<T, Context>(ctx, N, index, axis);

      auto grad_src = IndexSelect<T, Context>(ctx, out_grad, index, axis);

      *updates_grad = Divide<T, Context>(ctx, grad_src, N_src);
    }
  } else if (reducer == "mul" || reducer == "muliply") {
    auto zeros = Full<T, Context>(ctx, vectorize(out_grad.dims()), 0);
    auto ones = Full<T, Context>(ctx, vectorize(out_grad.dims()), 1);
    if (x_grad) {
      auto mask = Equal<T, Context>(ctx, x, zeros);
      auto masked_self = Where<T, Context>(ctx, mask, ones, x);

      auto masked_self_result = Scatter<T, Context>(
          ctx, x, index, new_source, false, axis, reducer, include_self);

      auto grad_mul_masked_self_result =
          Multiply<T, Context>(ctx, out_grad, masked_self_result);
      *x_grad =
          Divide<T, Context>(ctx, grad_mul_masked_self_result, masked_self);
    }

    if (updates_grad) {
      auto src_ones = Full<T, Context>(ctx, vectorize(new_source.dims()), 1);
      auto src_zeros = Full<T, Context>(ctx, vectorize(new_source.dims()), 1);
      auto src_zero = Equal<T, Context>(ctx, new_source, src_zeros);
      auto src_zero_t = Cast<bool, Context>(ctx, src_zero, x.dtype());

      auto src_num_zeros_inner =
          IndexAdd<T, Context>(ctx, zeros, new_index, src_zero_t, axis);

      auto src_num_zeros =
          IndexSelect<T, Context>(ctx, src_num_zeros_inner, index, axis);

      auto src_num_zeros_equal_one =
          Equal<T, Context>(ctx, src_num_zeros, src_ones);

      auto src_single_zero_bool =
          BitwiseAnd<bool, Context>(ctx, src_zero, src_num_zeros_equal_one);

      auto masked_src =
          Where<T, Context>(ctx, src_single_zero_bool, src_ones, new_source);

      auto masked_src_result = Scatter<T, Context>(
          ctx, x, index, masked_src, false, axis, reducer, include_self);

      auto grad_mul_masked_src_result =
          Multiply<T, Context>(ctx, out_grad, masked_src_result);
      auto grad_mul_masked_src_result_index_select =
          IndexSelect<T, Context>(ctx, grad_mul_masked_src_result, index, axis);

      auto grad_mul_out = Multiply<T, Context>(ctx, out_grad, out);

      auto grad_mul_out_index_select =
          IndexSelect<T, Context>(ctx, grad_mul_out, index, axis);

      auto src_masked_fill_one =
          Where<T, Context>(ctx, src_zero, src_ones, new_source);
      auto where_2 = Divide<T, Context>(
          ctx, grad_mul_out_index_select, src_masked_fill_one);

      auto grad_src1 =
          Where<T, Context>(ctx,
                            src_single_zero_bool,
                            grad_mul_masked_src_result_index_select,
                            where_2);

      auto tmp_ones = Full<T, Context>(ctx, vectorize(src_num_zeros.dims()), 1);
      auto src_num_zeros_greater_one =
          GreaterThan<T, Context>(ctx, src_num_zeros, tmp_ones);
      auto src_num_zeros_greater_one_any =
          Any<bool, Context>(ctx, src_num_zeros_greater_one, {}, false);

      bool out_data = src_num_zeros_greater_one_any.template data<bool>()[0];
      if (out_data) {
        VLOG(3) << "index_reduce(): Double backward is unsupported for "
                   "new_source when "
                   ">1  zeros in new_source are scattered to the same position "
                   "in x";
        *updates_grad = grad_src1;
      } else {
        *updates_grad = grad_src1;
      }
    }

  } else if (reducer == "amin" || reducer == "amax") {
    auto value = IndexSelect<T, Context>(ctx, out, index, axis);

    auto self_is_result = Equal<T, Context>(ctx, x, out);
    auto self_is_result_t = Cast<bool, Context>(ctx, self_is_result, x.dtype());

    auto source_is_result = Equal<T, Context>(ctx, new_source, value);
    auto source_is_result_t =
        Cast<bool, Context>(ctx, source_is_result, x.dtype());

    auto N_to_distribute = IndexAdd<T, Context>(
        ctx, self_is_result_t, new_index, source_is_result_t, axis);

    auto grad_distributed = Divide<T, Context>(ctx, out_grad, N_to_distribute);

    if (x_grad) {
      *x_grad = Multiply<T, Context>(ctx, self_is_result_t, grad_distributed);
    }

    if (updates_grad) {
      auto src_grad_dist =
          IndexSelect<T, Context>(ctx, grad_distributed, index, axis);

      *updates_grad =
          Multiply<T, Context>(ctx, source_is_result_t, src_grad_dist);
    }
  } else if (reducer == "assign") {
    if (x_grad) {
      include_self = false;
    }

    if (updates_grad) {
      *updates_grad = IndexSelect<T, Context>(ctx, out_grad, index, axis);
    }
  }

  if (!include_self && x_grad) {
    auto self_dims = out_grad.dims();
    auto zeros = Full<T, Context>(ctx, vectorize(self_dims), 0);
    auto src_ones = Full<T, Context>(ctx, vectorize(new_source.dims()), 1);
    auto src_cnts = IndexAdd<T, Context>(ctx, zeros, new_index, src_ones, axis);
    auto mask = Equal<T, Context>(ctx, src_cnts, zeros);
    *x_grad = Where<T, Context>(ctx, mask, out_grad, zeros);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(scatter_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ScatterGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
