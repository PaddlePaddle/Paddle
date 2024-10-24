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

#include "paddle/phi/kernels/cum_maxmin_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

#ifdef _MSC_VER
template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type isnan_(T x) {
  return false;
}
template <typename T>
typename std::enable_if<!std::is_integral<T>::value, bool>::type isnan_(T x) {
  return std::isnan(x);
}
#else
template <typename T>
bool isnan_(T x) {
  return std::isnan(x);
}
#endif

template <typename T>
T compute_stride(T axis, phi::DDim dims) {
  T size = 1;
  for (T i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T1, typename T2, typename BinaryFunction>
void ComputeImp(const DenseTensor& x,
                DenseTensor* out,
                DenseTensor* indices,
                int64_t axis) {
  int ndims = x.dims().size();
  int finished = 0;
  std::vector<int64_t> counter(ndims, 0);
  const T1* x_data = x.data<T1>();
  T1* values_data = out->data<T1>();
  T2* indices_data = indices->data<T2>();
  int64_t x_stride = compute_stride<int64_t>(axis, x.dims());
  int64_t values_stride = compute_stride<int64_t>(axis, out->dims());
  int64_t indices_stride = compute_stride<int64_t>(axis, indices->dims());
  auto x_dim_vec = common::vectorize<int>(x.dims());
  int x_dim_size = x_dim_vec[axis];
  BinaryFunction op;

  while (!finished) {
    T1 max = *reinterpret_cast<const T1*>(x_data);
    int idx = 0;
    for (int i = 0; i < x_dim_size; i++) {
      T1 curr_elem = *reinterpret_cast<const T1*>(&x_data[i * x_stride]);
      if (isnan_(curr_elem) || (!isnan_(max) && op(curr_elem, max))) {
        max = curr_elem;
        idx = i;
      }
      values_data[i * values_stride] = max;
      indices_data[i * indices_stride] = idx;
    }
    if (ndims == 1) break;
    for (int dim_i = 0; dim_i < ndims; dim_i++) {
      if (dim_i == axis) {
        if (dim_i == (ndims - 1)) {
          finished = 1;
          break;
        }
        continue;
      }
      int64_t x_stride_ = compute_stride<int64_t>(dim_i, x.dims());
      int64_t values_stride_ = compute_stride<int64_t>(dim_i, out->dims());
      int64_t indices_stride_ = compute_stride<int64_t>(dim_i, indices->dims());
      counter[dim_i]++;
      x_data += x_stride_;
      values_data += values_stride_;
      indices_data += indices_stride_;
      if (counter[dim_i] == x_dim_vec[dim_i]) {
        if (dim_i == ndims - 1) {
          finished = 1;
          break;
        } else {
          x_data -= counter[dim_i] * x_stride_;
          values_data -= counter[dim_i] * values_stride_;
          indices_data -= counter[dim_i] * indices_stride_;
          counter[dim_i] = 0;
        }
      } else {
        break;
      }
    }
  }
}

template <typename T1, typename T2, typename BinaryFunction, typename Context>
void ScanWithIndicesKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           int axis,
                           DenseTensor* out,
                           DenseTensor* indices) {
  dev_ctx.template Alloc<T1>(out);
  dev_ctx.template Alloc<T2>(indices);

  // For 0D Tensor
  if (x.numel() == 1) {
    auto raw_dims = out->dims();
    phi::Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    phi::funcs::SetConstant<Context, T2> set_zero;
    set_zero(dev_ctx, indices, static_cast<T2>(0.0));
    out->Resize(raw_dims);
    indices->Resize(raw_dims);
    return;
  }
  auto out_dims = out->dims();

  PADDLE_ENFORCE_EQ(
      axis < out_dims.size() && axis >= (0 - out_dims.size()),
      true,
      common::errors::OutOfRange(
          "Attr(axis) is out of range, It's expected "
          "to be in range of [-%d, %d]. But received Attr(axis) = %d.",
          out_dims.size(),
          out_dims.size() - 1,
          axis));

  if (axis < 0) {
    axis = axis + out_dims.size();
  }
  ComputeImp<T1, T2, BinaryFunction>(x, out, indices, axis);
}

template <typename T, typename Context>
void CummaxKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int axis,
                  DataType dtype,
                  DenseTensor* out,
                  DenseTensor* indices) {
  if (dtype == DataType::INT32) {
    ScanWithIndicesKernel<T, int32_t, std::greater_equal<T>, Context>(
        dev_ctx, x, axis, out, indices);
  } else if (dtype == DataType::INT64) {
    ScanWithIndicesKernel<T, int64_t, std::greater_equal<T>, Context>(
        dev_ctx, x, axis, out, indices);
  }
}

template <typename T, typename Context>
void CumminKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int axis,
                  DataType dtype,
                  DenseTensor* out,
                  DenseTensor* indices) {
  if (dtype == DataType::INT32) {
    ScanWithIndicesKernel<T, int32_t, std::less_equal<T>, Context>(
        dev_ctx, x, axis, out, indices);
  } else if (dtype == DataType::INT64) {
    ScanWithIndicesKernel<T, int64_t, std::less_equal<T>, Context>(
        dev_ctx, x, axis, out, indices);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cummax,
                   CPU,
                   ALL_LAYOUT,
                   phi::CummaxKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}

PD_REGISTER_KERNEL(cummin,
                   CPU,
                   ALL_LAYOUT,
                   phi::CumminKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}
