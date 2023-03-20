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

#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/reduce.h"

namespace phi {

template <typename TI, typename TO, typename Context>
void SumRawToINT64KernelImpl(const Context& dev_ctx,
                             const DenseTensor& x,
                             const IntArray& dims,
                             bool keep_dim,
                             bool reduce_all,
                             DenseTensor* out) {
  DenseTensor tmp_x;
  tmp_x.Resize(x.dims());
  TO* tmp_x_data = dev_ctx.template Alloc<TO>(&tmp_x);
  int r1 = xpu::cast<TI, TO>(
      dev_ctx.x_context(), x.data<TI>(), tmp_x_data, x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r1, "cast");

  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  using XPUType = typename XPUTypeTrait<TO>::Type;
  auto f = [](xpu::Context* ctx,
              const TO* x,
              TO* y,
              const std::vector<int>& xdims,
              const std::vector<int>& reduce_dims) {
    return xpu::reduce_sum<XPUType>(ctx,
                                    reinterpret_cast<const XPUType*>(x),
                                    reinterpret_cast<XPUType*>(y),
                                    xdims,
                                    reduce_dims);
  };
  int r2 = XPUReduce<Context, TO>(
      dev_ctx, tmp_x, dims.GetData(), keep_dim, reduce_all, out, f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r2, "reduce_sum");
}

template <typename T, typename Context>
void SumRawKernelImpl(const Context& dev_ctx,
                      const DenseTensor& x,
                      const IntArray& dims,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* ctx,
              const T* x,
              T* y,
              const std::vector<int>& xdims,
              const std::vector<int>& reduce_dims) {
    return xpu::reduce_sum<XPUType>(ctx,
                                    reinterpret_cast<const XPUType*>(x),
                                    reinterpret_cast<XPUType*>(y),
                                    xdims,
                                    reduce_dims);
  };
  int r = XPUReduce<Context, T>(
      dev_ctx, x, dims.GetData(), keep_dim, reduce_all, out, f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
}

template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DataType out_dtype,
                  DenseTensor* out) {
  // reduce_sum only support bool/int32 input transformed to int64.
  if (out_dtype == DataType::UNDEFINED && out->dtype() != x.dtype() &&
      out->dtype() == phi::DataType::INT64) {
    out_dtype = out->dtype();
    if (x.dtype() == phi::DataType::INT32) {
      SumRawToINT64KernelImpl<int32_t, int64_t, Context>(
          dev_ctx, x, dims, keep_dim, reduce_all, out);
    } else if (x.dtype() == phi::DataType::BOOL) {
      SumRawToINT64KernelImpl<bool, int64_t, Context>(
          dev_ctx, x, dims, keep_dim, reduce_all, out);
    } else {
      PADDLE_THROW(
          errors::Unimplemented("Only support transform int32/bool input to "
                                "int64 output in reduce_usm op"));
    }
  } else {
    SumRawKernelImpl<T, Context>(dev_ctx, x, dims, keep_dim, reduce_all, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sum_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::SumRawKernel,
                   float,
                   phi::dtype::float16,
                   int8_t,
                   int,
                   int64_t) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
