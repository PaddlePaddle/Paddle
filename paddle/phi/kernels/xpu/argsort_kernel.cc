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

#include "paddle/phi/kernels/argsort_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename TID>
static inline void xpu_argsort(xpu::Context* ctx,
                               const T* input_data,
                               T* output_data,
                               TID* indices_data,
                               int64_t m,
                               int64_t n,
                               bool descending,
                               bool stable) {
  int ret;
  if (stable) {
    ret = xpu::stable_sort(
        ctx, input_data, output_data, indices_data, m, n, descending);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "stable_sort");
  } else {
    ret =
        xpu::sort(ctx, input_data, output_data, indices_data, m, n, descending);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "sort");
  }
}

template <typename T>
static inline void xpu_transpose(xpu::Context* ctx,
                                 const T* x,
                                 T* y,
                                 const std::vector<int64_t>& xshape,
                                 const std::vector<int64_t>& permute) {
  int ret = xpu::transpose(ctx, x, y, xshape, permute);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "transpose");
}

template <typename T>
struct XPUArgsort {
  void operator()(xpu::Context* ctx,
                  const T* input_data,
                  T* output_data,
                  int64_t* indices_data,
                  const std::vector<int64_t>& data_shape,
                  const std::vector<int64_t>& permute,
                  bool descending,
                  bool stable) {
    xpu::ctx_guard RAII_GUARD(ctx);
    int64_t m = data_shape[0] * data_shape[2];
    int64_t n = data_shape[1];
    int64_t len = data_shape[0] * data_shape[1] * data_shape[2];
    std::vector<int64_t> trans_data_shape{
        data_shape[0], data_shape[2], data_shape[1]};

    T* input_data_trans = RAII_GUARD.alloc_l3_or_gm<T>(len);
    T* output_data_trans = RAII_GUARD.alloc_l3_or_gm<T>(len);
    int64_t* indices_data_trans = RAII_GUARD.alloc_l3_or_gm<int64_t>(len);

    xpu_transpose(ctx, input_data, input_data_trans, data_shape, permute);
    xpu_argsort(ctx,
                input_data_trans,
                output_data_trans,
                indices_data_trans,
                m,
                n,
                descending,
                stable);
    xpu_transpose(
        ctx, output_data_trans, output_data, trans_data_shape, permute);
    xpu_transpose(
        ctx, indices_data_trans, indices_data, trans_data_shape, permute);
  }
};

template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   int axis,
                   bool descending,
                   bool stable,
                   DenseTensor* output,
                   DenseTensor* indices) {
  auto in_dims = input.dims();
  auto rank = in_dims.size();

  if (input.numel() == 0) {
    output->Resize(in_dims);
    indices->Resize(in_dims);
    dev_ctx.template Alloc<T>(output);
    dev_ctx.template Alloc<int64_t>(indices);
    return;
  }

  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  int64_t n = in_dims[axis];

  auto input_data = input.data<T>();
  auto output_data = dev_ctx.template Alloc<T>(output);
  auto indices_data = dev_ctx.template Alloc<int64_t>(indices);

  if (rank == 0) {
    phi::Copy<Context>(dev_ctx, input, dev_ctx.GetPlace(), false, output);
    phi::funcs::set_constant(dev_ctx, indices, static_cast<int64_t>(0));
    return;
  }

  int64_t len_before = common::product(common::slice_ddim(in_dims, 0, axis));
  int64_t len_after =
      common::product(common::slice_ddim(in_dims, axis + 1, in_dims.size()));
  std::vector<int64_t> permute_vec{0, 2, 1};
  std::vector<int64_t> data_shape{len_before, n, len_after};

  using XPUType = typename XPUTypeTrait<T>::Type;

  XPUArgsort<XPUType>()(dev_ctx.x_context(),
                        reinterpret_cast<const XPUType*>(input_data),
                        reinterpret_cast<XPUType*>(output_data),
                        indices_data,
                        data_shape,
                        permute_vec,
                        descending,
                        stable);
}
}  // namespace phi

PD_REGISTER_KERNEL(argsort,
                   XPU,
                   ALL_LAYOUT,
                   phi::ArgsortKernel,
                   float,
                   int,
                   int64_t,
                   phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
