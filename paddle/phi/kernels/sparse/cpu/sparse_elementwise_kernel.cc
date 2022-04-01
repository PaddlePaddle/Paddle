/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LTCENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS TS" BASTS,
WTTHOUT WARRANTTES OR CONDTTTONS OF ANY KTND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/sparse_elementwise_kernel.h"

#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_elementwise_base.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace sparse {

template <typename Functor, typename T, typename Context>
void ElementWiseCsrKernel(const Context& dev_ctx,
                          const SparseCsrTensor& x,
                          const SparseCsrTensor& y,
                          Functor func,
                          SparseCsrTensor* out) {
  PADDLE_ENFORCE_EQ(x.dims(), y.dims(), "x.dim and y.dim should be same");
  const DDim& x_dims = x.dims();
  const auto& n_row=x_dims[0];
  const auto& n_col=x_dims[1];
  const auto& x_crows = x.non_zero_crows();
  const auto& x_cols = x.non_zero_cols();
  const auto& x_values = x.non_zero_elements();
  const auto* x_crows_data = x_crows.data<int64_t>();
  const auto* x_cols_data = x_cols.data<int64_t>();
  const auto* x_values_data = x_values.data<T>();
  const auto place = dev_ctx.GetPlace();

  std::vector<int64_t>  next(n_col,-1);
  std::vector<T> A_row(n_col, 0);
  std::vector<T> B_row(n_col, 0);

  int64_t nnz = 0;
  std::vector<int64_t> Cp;
  std::vector<int64_t> Cj;
  std::vector<int64_t> Cx;
  Cp[0] = 0;

  for(int64_t i = 0; i < n_row; i++){
    int64_t head   = -2;
    int64_t length =  0;

    //add a row of A to A_row
    int64_t i_start = Ap[i];
    int64_t i_end   = Ap[i+1];
    for(int64_t jj = i_start; jj < i_end; jj++){
      int64_t j = Aj[jj];

      A_row[j] += Ax[jj];

      if(next[j] == -1){
        next[j] = head;
        head = j;
        length++;
      }
    }

    //add a row of B to B_row
    i_start = Bp[i];
    i_end   = Bp[i+1];
    for(int64_t jj = i_start; jj < i_end; jj++){
      int64_t j = Bj[jj];

      B_row[j] += Bx[jj];

      if(next[j] == -1){
        next[j] = head;
        head = j;
        length++;
      }
    }


    // scan through columns where A or B has
    // contributed a non-zero entry
    for(int64_t jj = 0; jj < length; jj++){
      T result = op(A_row[head], B_row[head]);

      if(result != 0){
        Cj[nnz] = head;
        Cx[nnz] = result;
        nnz++;
      }

      int64_t temp = head;
      head = next[head];

      next[temp]  = -1;
      A_row[temp] =  0;
      B_row[temp] =  0;
    }

    Cp[i + 1] = nnz;
  }




  DenseTensorMeta crows_meta(
      DataType::INT64, x.non_zero_crows().dims(), DataLayout::NCHW);
  DenseTensorMeta cols_meta(
      DataType::INT64, x.non_zero_cols().dims(), DataLayout::NCHW);
  DenseTensorMeta values_meta(
      paddle::experimental::CppTypeToDataType<T>::Type(),
      x.non_zero_elements().dims(),
      DataLayout::NCHW);

  phi::DenseTensor out_crows = phi::Empty(dev_ctx, std::move(crows_meta));
  phi::DenseTensor out_cols = phi::Empty(dev_ctx, std::move(cols_meta));
  phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));

  auto* out_crows_data = out_crows.mutable_data<int64_t>(place);
  auto* out_cols_data = out_cols.mutable_data<int64_t>(place);

  std::memcpy(
      out_crows_data, x_crows_data, sizeof(int64_t) * x_crows.dims()[0]);
  std::memcpy(out_cols_data, x_cols_data, sizeof(int64_t) * x_cols.dims()[0]);


  template <typename Functor, typename T, typename OutType = T>
void ElementwiseComputeCoo(const CPUContext& dev_ctx,
                           const SparseCooTensor& x,
                           const SparseCooTensor& y,
                           int axis,
                           Functor func,
                           SparseCooTensor* out) {
  PADDLE_ENFORCE_EQ(x.dims(), y.dims(), "x.dim and y.dim should be same");
  dev_ctx.Alloc<OutType>(out);

  //  std::plus<T> plus_func;
  //  plus_func(x.data<T>(), y.data<T>(), out->data<OutType>(), x.numel());
  //  auto x_dims = x.dims();
  //  auto y_dims = y.dims();
  //  bool is_xsize_larger = true;
  //  int max_dim = x_dims.size();
  //  if (x_dims.size() < y_dims.size()) {
  //    is_xsize_larger = false;
  //    max_dim = y_dims.size();
  //  }
  funcs::sparse::TransformFunctor<Functor, T, CPUContext, OutType> functor(
      x, y, out, dev_ctx, func, is_xsize_larger);
  if (x_dims == y_dims) {
    functor.Run();
    return;
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis,
                    max_dim,
                    errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim,
                        axis));

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimed = TrimTrailingSingularDims(y_dims);
    axis_trim = (y_dims_trimed.size() == 0) ? x_dims.size() : axis;
    GetMidDims(x_dims,
               y_dims_trimed,
               axis_trim,
               &pre,
               &n,
               &post,
               &is_run_common_broadcast);
  } else {
    auto x_dims_trimed = TrimTrailingSingularDims(x_dims);
    axis_trim = (x_dims_trimed.size() == 0) ? y_dims.size() : axis;
    GetMidDims(y_dims,
               x_dims_trimed,
               axis_trim,
               &pre,
               &n,
               &post,
               &is_run_common_broadcast);
  }
  // special case for common implementation.
  // case 1: x=[2,3,1,5], y=[2,1,4,1]
  // case 2: x=[2,3,4], y=[1,1,4]
  if (is_run_common_broadcast == 1) {
    CommonElementwiseBroadcastForward<Functor, T, OutType>(
        dev_ctx, x, y, out, x_dims, y_dims, func, axis, is_xsize_larger);
    return;
  }

  if (post == 1) {
    functor.RunRowWise(n, pre);
    return;
  } else {
    functor.RunMidWise(n, pre, post);
    return;
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_elementwise_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseCsrKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
