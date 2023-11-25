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

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/fused_gemm_epilogue.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedGemmEpilogueGradKernel(
    const Context& dev_ctx,
    const DenseTensor& d_out,
    const DenseTensor& x,
    const DenseTensor& y,
    const paddle::optional<DenseTensor>& reserve_space,
    const bool trans_x,
    const bool trans_y,
    const std::string& activation,
    DenseTensor* d_x,
    DenseTensor* d_y,
    DenseTensor* d_bias) {
#if CUDA_VERSION < 11060
  PADDLE_THROW(phi::errors::Unimplemented(
      "The fused_gemm_epilogue operator only support CUDA 11.6 "
      "or higher version."));
#endif

  // (M * K) * (K * N)
  auto x_mat_dims =
      phi::flatten_to_2d(x.dims(), trans_x ? 1 : x.dims().size() - 1);
  int64_t M = trans_x ? x_mat_dims[1] : x_mat_dims[0];
  int64_t K = trans_y ? y.dims()[1] : y.dims()[0];
  int64_t N = trans_y ? y.dims()[0] : y.dims()[1];

  VLOG(6) << "x.shape={" << x.dims() << "}, y.shape={" << y.dims()
          << "}, dout.shape={" << d_out.dims() << "}, M=" << M << ", N=" << N
          << ", K=" << K << ", trans_x=" << trans_x << ", trans_y=" << trans_y
          << ", activation=" << activation
          << ", reserve_space=" << reserve_space.get_ptr();

  phi::funcs::ComputeFusedGemmEpilogueBackward<T>(dev_ctx,
                                                  &d_out,
                                                  &x,
                                                  &y,
                                                  reserve_space.get_ptr(),
                                                  M,
                                                  N,
                                                  K,
                                                  trans_x,
                                                  trans_y,
                                                  activation,
                                                  d_x,
                                                  d_y,
                                                  d_bias);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_gemm_epilogue_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGemmEpilogueGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
