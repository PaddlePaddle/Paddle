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

#include "paddle/phi/kernels/slice_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {

template <typename T, typename Context>
void SliceGradKernel(const Context& ctx,
                     const DenseTensor& input,
                     const DenseTensor& out_grad,
                     const std::vector<int64_t>& axes,
                     const IntArray& starts_t,
                     const IntArray& ends_t,
                     const std::vector<int64_t>& infer_flags,
                     const std::vector<int64_t>& decrease_axis,
                     DenseTensor* input_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  ctx.template Alloc<T>(input_grad);
  if (input_grad->numel() == 0) {
    return;
  }
  if (out_grad.numel() == 0) {
    phi::Full<T, XPUContext>(
        ctx,
        phi::IntArray(common::vectorize(input_grad->dims())),
        T(0),
        input_grad);
    return;
  }
  // Get the accurate attribute value of starts and ends
  std::vector<int64_t> starts = starts_t.GetData();
  std::vector<int64_t> ends = ends_t.GetData();

  const auto& in_dims = input.dims();
  int rank = in_dims.size();

  std::vector<int64_t> pad_left(rank);
  std::vector<int64_t> out_dims(rank);
  std::vector<int64_t> pad_right(rank);
  int64_t cnt = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    int64_t start = 0;
    int64_t end = in_dims[i];
    int64_t axis = cnt < static_cast<int64_t>(axes.size()) ? axes[cnt] : -1;
    if (axis == i) {
      bool zero_dim = false;
      funcs::normalize_interval(starts[cnt],
                                ends[cnt],
                                static_cast<int64_t>(1),
                                in_dims[i],
                                &start,
                                &end,
                                &zero_dim);
      cnt++;
    }

    pad_left[i] = start;
    out_dims[i] = end - start;
    pad_right[i] = in_dims[i] - out_dims[i] - pad_left[i];
  }

  int r =
      xpu::pad<XPUType>(ctx.x_context(),
                        reinterpret_cast<const XPUType*>(out_grad.data<T>()),
                        reinterpret_cast<XPUType*>(input_grad->data<T>()),
                        out_dims,
                        pad_left,
                        pad_right,
                        XPUType(0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
}

#ifdef PADDLE_WITH_XPU_FFT
template <>
void SliceGradKernel<phi::dtype::complex<float>, XPUContext>(
    const XPUContext& ctx,
    const DenseTensor& input,
    const DenseTensor& out_grad,
    const std::vector<int64_t>& axes,
    const IntArray& starts_t,
    const IntArray& ends_t,
    const std::vector<int64_t>& infer_flags,
    const std::vector<int64_t>& decrease_axis,
    DenseTensor* input_grad) {
  using T = phi::dtype::complex<float>;
  ctx.template Alloc<T>(input_grad);
  if (input_grad->numel() == 0) {
    return;
  }
  if (out_grad.numel() == 0) {
    phi::Full<T, XPUContext>(
        ctx,
        phi::IntArray(common::vectorize(input_grad->dims())),
        T(0),
        input_grad);
    return;
  }

  // Get the accurate attribute value of starts and ends
  std::vector<int64_t> starts = starts_t.GetData();
  std::vector<int64_t> ends = ends_t.GetData();

  const auto& in_dims = input.dims();
  int rank = in_dims.size();

  std::vector<int64_t> pad_left(rank);
  std::vector<int64_t> out_dims(rank);
  std::vector<int64_t> pad_right(rank);
  int64_t cnt = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    int64_t start = 0;
    int64_t end = in_dims[i];
    int64_t axis = cnt < static_cast<int64_t>(axes.size()) ? axes[cnt] : -1;
    if (axis == i) {
      bool zero_dim = false;
      funcs::normalize_interval(starts[cnt],
                                ends[cnt],
                                static_cast<int64_t>(1),
                                in_dims[i],
                                &start,
                                &end,
                                &zero_dim);
      cnt++;
    }

    pad_left[i] = start;
    out_dims[i] = end - start;
    pad_right[i] = in_dims[i] - out_dims[i] - pad_left[i];
  }

  // The current complex number implementation uses separate real/imaginary
  // parts,resulting in redundant operations and performance
  // penalties.Optimization should address this in future iterations.
  const DenseTensor real = Real<T, XPUContext>(ctx, out_grad);
  const DenseTensor imag = Imag<T, XPUContext>(ctx, out_grad);
  DenseTensor real_out, imag_out;
  real_out.Resize(input_grad->dims());
  imag_out.Resize(input_grad->dims());
  ctx.template Alloc<float>(&real_out);
  ctx.template Alloc<float>(&imag_out);
  int r = xpu::pad<float>(ctx.x_context(),
                          real.data<float>(),
                          real_out.data<float>(),
                          out_dims,
                          pad_left,
                          pad_right,
                          static_cast<float>(0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
  r = xpu::pad<float>(ctx.x_context(),
                      imag.data<float>(),
                      imag_out.data<float>(),
                      out_dims,
                      pad_left,
                      pad_right,
                      static_cast<float>(0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
  phi::ComplexKernel<float>(ctx, real_out, imag_out, input_grad);
}
#endif
}  // namespace phi

PD_REGISTER_KERNEL(slice_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SliceGradKernel,
                   float,
                   int,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::dtype::complex<float>,
#endif
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
}
