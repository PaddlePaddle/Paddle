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

#include "paddle/phi/kernels/bmm_kernel.h"
#include "paddle/phi/kernels/xpu/bmm_xpu_utils.h"
namespace phi {
template <typename T, typename Context>
void BmmKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (x.numel() == 0 || y.numel() == 0) {
    return;
  }
  bool trans_x = false;
  bool trans_y = false;

  auto x_dims = x.dims();
  auto y_dims = y.dims();

  PADDLE_ENFORCE_EQ(x_dims.size(),
                    3,
                    phi::errors::InvalidArgument(
                        "Input(X) of BmmOp must be 3-dimensional in BmmOp, "
                        "but received X's shape: [%s]",
                        x_dims));
  PADDLE_ENFORCE_EQ(y_dims.size(),
                    3,
                    phi::errors::InvalidArgument(
                        "Input(Y) of BmmOp must be 3-dimensional in BmmOp, "
                        "but received Y's shape: [%s].",
                        y_dims));
  PADDLE_ENFORCE_EQ(
      x_dims[0],
      y_dims[0],
      phi::errors::InvalidArgument(
          "Input(X) and Input(Y) must have the same batch size in BmmOp, "
          "but received X's batch size: [%s],"
          "Y's batch size [%s]",
          x_dims[0],
          y_dims[0]));
  PADDLE_ENFORCE_EQ(
      x_dims[2],
      y_dims[1],
      phi::errors::InvalidArgument(
          "Input(X)'s width must be equal with Input(Y)'s height in BmmOp,"
          "but receive X's width: [%s],"
          "Y's height: [%s].",
          x_dims[2],
          y_dims[1]));

  xpu::Context* xpu_ctx = dev_ctx.x_context();
  if (std::is_same<phi::dtype::float16, T>::value) {
    MatMulXPUFunction<T, int16_t>(x, y, out, trans_x, trans_y, xpu_ctx);
  } else {
    if (std::getenv("XPU_PADDLE_FC_INT32") != nullptr) {
      MatMulXPUFunction<T, int32_t>(x, y, out, trans_x, trans_y, xpu_ctx);
    } else if (std::getenv("XPU_PADDLE_FC_LOCAL_INT16") != nullptr) {
      MatMulXPUFunction<T, float>(x, y, out, trans_x, trans_y, xpu_ctx);
    } else {
      MatMulXPUFunction<T, int16_t>(x, y, out, trans_x, trans_y, xpu_ctx);
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    bmm, XPU, ALL_LAYOUT, phi::BmmKernel, float, phi::dtype::float16) {}
