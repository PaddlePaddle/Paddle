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
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/fused_gemm_epilogue.h"

namespace phi {
namespace fusion {
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060) || \
    defined(PADDLE_WITH_HIP)
template <typename T>
phi::funcs::MatmulFusedType GetFwdFusedEpilogueType(
    const phi::GPUContext& ctx,
    const std::string& activation,
    phi::DenseTensor* reserve_space) {
  using FusedType = phi::funcs::MatmulFusedType;

  FusedType fused_type = FusedType::kMatmulBias;
  if (activation != "none") {
    if (activation == "relu") {
      if (reserve_space == nullptr) {
        fused_type = FusedType::kMatmulBiasRelu;
      } else {
#ifdef PADDLE_WITH_HIP
        PADDLE_THROW(
            common::errors::Unimplemented("kMatmulBiasReluWithReservedData is "
                                          "not supported on HIP platform."));
#else
        fused_type = FusedType::kMatmulBiasReluWithReservedData;
        reserve_space->Resize({phi::product(reserve_space->dims())});
        ctx.template Alloc<bool>(reserve_space);
#endif
      }
    } else if (activation == "gelu") {
      if (reserve_space == nullptr) {
        fused_type = FusedType::kMatmulBiasGelu;
      } else {
        fused_type = FusedType::kMatmulBiasGeluWithReservedData;
        int64_t reserve_size = sizeof(T) * phi::product(reserve_space->dims());
        ctx.template Alloc<T>(reserve_space, reserve_size);
      }
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "fused_gemm_epilogue's activate should be one of {none, relu, gelu},"
          " but received %s, please check",
          activation));
    }
  }
  return fused_type;
}
#endif

template <typename T, typename Context>
void FusedGemmEpilogueKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& bias,
                             const bool trans_x,
                             const bool trans_y,
                             const std::string& activation,
                             DenseTensor* out,
                             DenseTensor* reserve_space) {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION < 11060
  PADDLE_THROW(common::errors::Unimplemented(
      "The fused_gemm_epilogue operator only support CUDA 11.6 "
      "or higher version."));
#endif
#if (defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060) || \
    defined(PADDLE_WITH_HIP)

  dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));
  // (M * K) * (K * N)
  auto x_mat_dims =
      phi::flatten_to_2d(x.dims(), trans_x ? 1 : x.dims().size() - 1);
  int64_t M = trans_x ? x_mat_dims[1] : x_mat_dims[0];
  int64_t K = trans_y ? y.dims()[1] : y.dims()[0];
  int64_t N = trans_y ? y.dims()[0] : y.dims()[1];

  auto fused_type =
      GetFwdFusedEpilogueType<T>(dev_ctx, activation, reserve_space);
  void* reserve_data = reserve_space ? reserve_space->data() : nullptr;

  VLOG(6) << "x.shape={" << x.dims() << "}, y.shape={" << y.dims()
          << "}, out.shape={" << out->dims() << "}, M=" << M << ", N=" << N
          << ", K=" << K << ", trans_x=" << trans_x << ", trans_y=" << trans_y
          << ", activation=" << activation << ", fused_type=" << fused_type
          << ", reserve_space=" << reserve_space;

  phi::funcs::LinearWithCublasLt<T>::Run(
      dev_ctx,
      &x,
      &y,
      out,
      static_cast<const void*>(bias.data<T>()),
      reserve_data,
      M,
      N,
      K,
      trans_x,
      trans_y,
      fused_type);
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_gemm_epilogue,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGemmEpilogueKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
