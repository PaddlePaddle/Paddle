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

#include "paddle/phi/kernels/index_select_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename Context, typename T, class Enable = void>
struct IndexSelectAdd {
  void operator()(const Context& ctx,
                  int slice_size,
                  const T* src_pointer,
                  const T* p_pointer,
                  T* dist_pointer) {
    for (int i = 0; i < slice_size; i++) {
      dist_pointer[i] = src_pointer[i] + p_pointer[i];
    }
  }
};

template <typename Context, typename T>
struct IndexSelectAdd<
    Context,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const Context& ctx,
                  int slice_size,
                  const T* src_pointer,
                  const T* p_pointer,
                  T* dist_pointer) {
    auto blas = phi::funcs::GetBlas<Context, T>(ctx);
    blas.VADD(slice_size, src_pointer, p_pointer, dist_pointer);
  }
};

template <typename Context, typename T, typename IndexT = int>
void IndexSelectGradInner(const Context& ctx,
                          const DenseTensor& out_grad,
                          const DenseTensor& index,
                          DenseTensor* x_grad,
                          int dim) {
  const T* input_data = out_grad.data<T>();
  const IndexT* index_data = index.data<IndexT>();

  const T* p_output = x_grad->mutable_data<T>(ctx.GetPlace());
  //  const T* p_output = ctx.template Alloc<T>(x_grad);
  T* out_data = x_grad->mutable_data<T>(ctx.GetPlace());
  //  T* out_data = ctx.template Alloc<T>(x_grad);

  auto input_dim = out_grad.dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = x_grad->dims();

  //  auto& dev_ctx = context.template device_context<DeviceContext>();
  phi::funcs::SetConstant<Context, T> set_constant;
  set_constant(ctx, x_grad, static_cast<T>(0.0));

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
      IndexSelectAdd<Context, T> index_select_add;
      index_select_add(ctx, slice_size, src, p_out, dst);
    }
  }
  x_grad->Resize(output_dim);
}

template <typename T, typename Context>
void IndexSelectGradKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& index,
                           const DenseTensor& out_grad,
                           int dim,
                           DenseTensor* x_grad) {
  if (dim < 0) {
    dim += out_grad.dims().size();
  }
  const auto& index_type =
      paddle::framework::TransToProtoVarType(index.dtype());

  bool index_type_match =
      index_type == paddle::framework::proto::VarType::INT32 ||
      index_type == paddle::framework::proto::VarType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        paddle::framework::DataTypeToString(index_type),
                        paddle::framework::DataTypeToString(
                            paddle::framework::proto::VarType::INT32),
                        paddle::framework::DataTypeToString(
                            paddle::framework::proto::VarType::INT64)));

  if (index_type == paddle::framework::proto::VarType::INT32) {
    IndexSelectGradInner<Context, T, int>(ctx, out_grad, index, x_grad, dim);
  } else if (index_type == paddle::framework::proto::VarType::INT64) {
    IndexSelectGradInner<Context, T, int64_t>(
        ctx, out_grad, index, x_grad, dim);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_select_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexSelectGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
