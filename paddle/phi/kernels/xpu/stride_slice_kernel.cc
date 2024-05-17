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

#include "paddle/phi/kernels/strided_slice_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/strided_slice.h"
#include "paddle/phi/kernels/xpu/stride_slice_util.h"

namespace phi {

template <typename T, typename Context>
void StridedSliceRawKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const std::vector<int>& axes,
                           const IntArray& starts,
                           const IntArray& ends,
                           const IntArray& strides,
                           const std::vector<int>& infer_flags,
                           const std::vector<int>& decrease_axis,
                           DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  DDim in_dims = x.dims();

  auto starts_ = starts.GetData();
  auto ends_ = ends.GetData();
  auto strides_ = strides.GetData();

  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
  funcs::StridedSliceOutDims(starts_,
                             ends_,
                             strides_,
                             axes,
                             infer_flags,
                             in_dims,
                             decrease_axis,
                             out_dims_vector.data(),
                             axes.size(),
                             false);
  DDim out_dims(common::make_ddim(out_dims_vector));

  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  std::vector<int> xshape;
  std::vector<int> starts_in(in_dims.size(), 0);
  std::vector<int> ends_in;
  std::vector<int> strides_in(in_dims.size(), 1);

  for (int i = 0; i < in_dims.size(); ++i) {
    xshape.emplace_back(in_dims[i]);
    ends_in.emplace_back(in_dims[i]);
  }

  int num = axes.size();
  for (int i = 0; i < num; ++i) {
    int cur_axe = axes[i];
    int st = starts_[i];
    if (st > xshape[cur_axe]) {
      st = xshape[cur_axe] - 1;
    }
    if (st < 0) {
      st += xshape[cur_axe];
    }
    starts_in[cur_axe] = st;

    int end = ends_[i];
    if (end > xshape[cur_axe]) {
      end = xshape[cur_axe];
    }
    if (end < 0) {
      if (!(end == -1 && strides_[i] < 0)) {
        end = end + xshape[cur_axe];
        if (end < 0) {
          end = 0;
        }
      }
    }

    ends_in[cur_axe] = end;
    strides_in[cur_axe] = strides_[i];
  }

  if (is_strided_slice_special_case(xshape, starts_in, ends_in, strides_in)) {
    PADDLE_ENFORCE_EQ(
        x.numel(),
        out->numel() * 2,
        errors::PreconditionNotMet(
            "x.numel() should be equal to out->numel() * 2 in special case."));
    /*
     * sample input: [1 2 3 4 5 6 7 8 9 10]
     * starts = [0/1]
     * strides = [2]
     * sample output: [1 3 5 7 9] (last value in starts is 0)
     * sample output: [2 4 6 8 10] (last value in starts is 1)
     */
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    XPUType* x_transpose = RAII_GUARD.alloc_l3_or_gm<XPUType>(x.numel());
    /*
     * step 1: transpose, input shape is (x.numel/2, 2):
     * input:
     * [1 2
     *  3 4
     *  5 6
     *  7 8
     *  9 10]
     * after transpose:
     * [1 3 5 7 9
     *  2 4 6 8 10]
     */
    // int transpose(Context* ctx, const T* x, T* y, const std::vector<int>&
    // xshape, const std::vector<int>& permute)
    int r =
        xpu::transpose<XPUType>(dev_ctx.x_context(),
                                reinterpret_cast<const XPUType*>(x.data<T>()),
                                x_transpose,
                                {x.numel() / 2, 2},
                                {1, 0});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    // step 2: if starts from 0, use "first half" data as result, otherwise use
    // "second half".
    int offset = 0;
    if (starts_in.back() == 1) {
      offset = x.numel() / 2;
    }
    // int copy(Context* ctx, const T* x, T* y, int64_t len)
    r = xpu::copy<XPUType>(dev_ctx.x_context(),
                           x_transpose + offset,
                           reinterpret_cast<XPUType*>(out->data<T>()),
                           x.numel() / 2);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    return;
  }

  int r = xpu::strided_slice(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(x.data<T>()),
                             reinterpret_cast<XPUType*>(out->data<T>()),
                             xshape,
                             starts_in,
                             ends_in,
                             strides_in);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_slice");
}

}  // namespace phi
PD_REGISTER_KERNEL(strided_slice_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::StridedSliceRawKernel,
                   int,
                   int16_t,
                   int64_t,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
