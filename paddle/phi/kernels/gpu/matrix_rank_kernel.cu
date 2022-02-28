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

#ifndef PADDLE_WITH_HIP
// HIP not support cusolver
#include <algorithm>
#include <vector>
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/fluid/platform/dynload/cusolver.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
struct GreaterElementFunctor {
  HOSTDEVICE T operator()(const T a, const T b) const {
    if (a > b) {
      return a;
    } else {
      return b;
    }
  }
};

template <typename T, typename Context>
void MatrixRankKernel(const Context& ctx,
                      const DenseTensor& x,
                      bool hermitian,
                      bool use_default_tol,
                      DenseTensor* out) {
  auto& dev_ctx =
      context.template device_context<platform::CUDADeviceContext>();

  // const Tensor* x = context.Input<Tensor>("X");
  auto* x_data = x.data<T>();
  // auto* out = context.Output<Tensor>("Out");
  dev_ctx.template Alloc<int64_t>(out);
  // out->mutable_data<int64_t>(context.GetPlace());
  // bool hermitian = context.Attr<bool>("hermitian");

  auto dim_x = x.dims();
  auto dim_out = out->dims();
  int rows = dim_x[dim_x.size() - 2];
  int cols = dim_x[dim_x.size() - 1];
  int k = std::min(rows, cols);
  auto numel = x.numel();
  int batches = numel / (rows * cols);

  // bool use_default_tol = context.Attr<bool>("use_default_tol");
  const DenseTensor* atol_tensor = nullptr;
  DenseTensor temp_tensor;
  T rtol_T = 0;
  if (use_default_tol) {
    framework::TensorFromVector<T>(
        std::vector<T>{0}, context.device_context(), &temp_tensor);
    atol_tensor = &temp_tensor;
    rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
  } else if (context.HasInput("TolTensor")) {
    atol_tensor = context.Input<Tensor>("TolTensor");
  } else {
    framework::TensorFromVector<T>(std::vector<T>{context.Attr<float>("tol")},
                                   context.device_context(),
                                   &temp_tensor);
    atol_tensor = &temp_tensor;
  }

  // Must Copy X once, because the gesvdj will destory the content when exit.
  DenseTensor x_tmp;
  paddle::framework::TensorCopy(*x, context.GetPlace(), &x_tmp);
  auto info = memory::Alloc(dev_ctx, sizeof(int) * batches);
  int* info_ptr = reinterpret_cast<int*>(info->ptr());

  DenseTensor eigenvalue_tensor;
  eigenvalue_tensor->Resize(detail::GetEigenvalueDim(dim_x, k));
  auto* eigenvalue_data = dev_ctx.template Alloc(tol_tensor);
  // auto* eigenvalue_data = eigenvalue_tensor.mutable_data<T>(
  //   detail::GetEigenvalueDim(dim_x, k), context.GetPlace());
  if (hermitian) {
    SyevjBatched(
        dev_ctx, batches, rows, x_tmp.data<T>(), eigenvalue_data, info_ptr);
    phi::funcs::ForRange<Context> for_range(ctx, eigenvalue_tensor.numel());
    phi::funcs::AbsFunctor<T> functor(
        eigenvalue_data, eigenvalue_data, eigenvalue_tensor.numel());
    for_range(functor);
  } else {
    DenseTensor U, VH;
    U->Resize(detail::GetUDDim(dim_x, k));
    auto* u_data = dev_ctx.template Alloc(tol_tensor);
    // auto* u_data =
    //     U.mutable_data<T>(detail::GetUDDim(dim_x, k), context.GetPlace());
    // auto* vh_data =
    //     VH.mutable_data<T>(detail::GetVHDDim(dim_x, k), context.GetPlace());
    U->Resize(detail::GetVHDDim(dim_x, k));
    auto* vh_data = dev_ctx.template Alloc(tol_tensor);
    GesvdjBatched(dev_ctx,
                  batches,
                  cols,
                  rows,
                  k,
                  x_tmp.data<T>(),
                  vh_data,
                  u_data,
                  eigenvalue_data,
                  info_ptr,
                  1);
  }

  auto dito_T =
      math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext, T>(
          context);
  std::vector<int> max_eigenvalue_shape =
      phi::vectorize<int>(detail::RemoveLastDim(eigenvalue_tensor.dims()));
  DenseTensor max_eigenvalue_tensor =
      dito_T.ReduceMax(eigenvalue_tensor, max_eigenvalue_shape);
  DenseTensor temp_rtol_tensor;
  framework::TensorFromVector<T>(
      std::vector<T>{rtol_T}, context.device_context(), &temp_rtol_tensor);
  DenseTensor rtol_tensor = dito_T.Mul(temp_rtol_tensor, max_eigenvalue_tensor);
  DenseTensor tol_tensor;
  // tol_tensor.mutable_data<T>(dim_out, context.GetPlace());
  tol_tensor->Resize(dim_out);
  dev_ctx.template Alloc(tol_tensor);

  ElementwiseComputeEx<GreaterElementFunctor<T>, phi::CUDADeviceContext, T, T>(
      context,
      atol_tensor,
      &rtol_tensor,
      -1,
      GreaterElementFunctor<T>(),
      &tol_tensor);

  tol_tensor.Resize(detail::NewAxisDim(tol_tensor.dims(), 1));

  DenseTensor compare_result;
  compare_result->Resize(detail::NewAxisDim(dim_out, k));
  dev_ctx.template Alloc(compare_result);
  // compare_result.mutable_data<int64_t>(detail::NewAxisDim(dim_out, k),
  //                                   context.GetPlace());
  int axis = -1;
  ElementwiseComputeEx<GreaterThanFunctor<T, int64_t>,
                       phi::CUDADeviceContext,
                       T,
                       int64_t>(context,
                                &eigenvalue_tensor,
                                &tol_tensor,
                                axis,
                                GreaterThanFunctor<T, int64_t>(),
                                &compare_result);
  auto dito_int =
      math::DeviceIndependenceTensorOperations<phi::CUDADeviceContext, int64_t>(
          context);
  std::vector<int> result_shape = phi::vectorize<int>(dim_out);
  Tensor result = dito_int.ReduceSum(compare_result, result_shape);
  out->ShareDataWith(result);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    matrix_rank, GPU, ALL_LAYOUT, phi::MatrixRankKernel, float, double) {}

#endif  // not PADDLE_WITH_HIP
