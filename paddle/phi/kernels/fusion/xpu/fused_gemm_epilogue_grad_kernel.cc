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

#include "glog/logging.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/scope_guard.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

#include "paddle/phi/kernels/funcs/fused_gemm_epilogue_xpu.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedGemmEpilogueXPUGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    const paddle::optional<DenseTensor>& reserve_space,
    const DenseTensor& out_grad,
    const bool trans_x,
    const bool trans_y,
    const std::string& activation_grad,
    DenseTensor* x_grad,
    DenseTensor* y_grad,
    DenseTensor* bias_grad) {
  // (M * K) * (K * N)
  auto x_mat_dims =
      phi::flatten_to_2d(x.dims(), trans_x ? 1 : x.dims().size() - 1);
  int64_t M = trans_x ? x_mat_dims[1] : x_mat_dims[0];
  int64_t K = trans_y ? y.dims()[1] : y.dims()[0];
  int64_t N = trans_y ? y.dims()[0] : y.dims()[1];

  VLOG(6) << "x.shape={" << x.dims() << "}, y.shape={" << y.dims()
          << "}, dout.shape={" << out_grad.dims() << "}, M=" << M << ", N=" << N
          << ", K=" << K << ", trans_x=" << trans_x << ", trans_y=" << trans_y
          << ", activation_grad=" << activation_grad
          << ", reserve_space=" << reserve_space.get_ptr();

  phi::funcs::ComputeFusedGemmEpilogueBackwardXPU<T>(dev_ctx,
                                                     &out_grad,
                                                     &x,
                                                     &y,
                                                     reserve_space.get_ptr(),
                                                     M,
                                                     N,
                                                     K,
                                                     trans_x,
                                                     trans_y,
                                                     activation_grad,
                                                     x_grad,
                                                     y_grad,
                                                     bias_grad);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_gemm_epilogue_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGemmEpilogueXPUGradKernel,
                   float,
                   phi::dtype::float16) {}
