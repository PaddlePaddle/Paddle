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

#include "paddle/phi/kernels/weight_only_matmul_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#if defined(PADDLE_WITH_CUTLASS)
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#endif

namespace phi {

template <typename T, typename Context>
void WeightOnlyMatmulKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            const DenseTensor& weight_scale,
                            DenseTensor* out) {
#if defined(PADDLE_WITH_CUTLASS)
  dev_ctx.template Alloc<T>(out);
  const auto x_dims = x.dims();
  const auto w_dims = weight.dims();
  int n = weight_scale.dims()[0];
  int quant_bit = 0;
  if (n % w_dims[0] == 0) {
    quant_bit = w_dims[0] * 8 / n;
  } else {
    errors::InvalidArgument(
        "w_dims[0] must be divisible by weight_scale.dims()[0]");
  }

  int k = w_dims[1];
  int m = x.numel() / k;
  switch (quant_bit) {
    case 8: {
      auto mixed_gemm_runner =
          CutlassFpAIntBGemmRunner<typename PDDataTypeTraits<T>::DataType,
                                   uint8_t>();
      int mixgemm_max_size = std::max(n, k);
      DenseTensor mixgemm_workspace;
      int64_t mixgemm_workspace_size_bytes = mixed_gemm_runner.getWorkspaceSize(
          m, mixgemm_max_size, mixgemm_max_size);

      mixgemm_workspace.Resize({mixgemm_workspace_size_bytes});
      dev_ctx.template Alloc<uint8_t>(&mixgemm_workspace);
      char* mixgemm_workspace_data =
          reinterpret_cast<char*>(mixgemm_workspace.data<uint8_t>());
      mixed_gemm_runner.gemm(
          reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
              x.data<T>()),
          reinterpret_cast<const uint8_t*>(weight.data<int8_t>()),
          reinterpret_cast<const float*>(weight_scale.data<float>()),
          reinterpret_cast<typename PDDataTypeTraits<T>::DataType*>(
              out->data<T>()),
          m,
          n,
          k,
          mixgemm_workspace_data,
          mixgemm_workspace_size_bytes,
          dev_ctx.stream());
    } break;
    case 4: {
      auto mixed_gemm_runner =
          CutlassFpAIntBGemmRunner<typename PDDataTypeTraits<T>::DataType,
                                   cutlass::uint4b_t>();
      int mixgemm_max_size = std::max(n, k);
      DenseTensor mixgemm_workspace;
      int64_t mixgemm_workspace_size_bytes = mixed_gemm_runner.getWorkspaceSize(
          m, mixgemm_max_size, mixgemm_max_size);

      mixgemm_workspace.Resize({mixgemm_workspace_size_bytes});
      dev_ctx.template Alloc<uint8_t>(&mixgemm_workspace);
      char* mixgemm_workspace_data =
          reinterpret_cast<char*>(mixgemm_workspace.data<uint8_t>());
      mixed_gemm_runner.gemm(
          reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
              x.data<T>()),
          reinterpret_cast<const cutlass::uint4b_t*>(weight.data<int8_t>()),
          reinterpret_cast<const float*>(weight_scale.data<float>()),
          reinterpret_cast<typename PDDataTypeTraits<T>::DataType*>(
              out->data<T>()),
          m,
          n,
          k,
          mixgemm_workspace_data,
          mixgemm_workspace_size_bytes,
          dev_ctx.stream());
    } break;
    default:
      PADDLE_THROW(errors::Unimplemented(
          "Quant_bits (%d) is not supported when gemm ", quant_bit));
      break;
  }

#else
  LOG(ERROR) << "Please compile with cutlass to EnableUseCutlass()";
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(weight_only_matmul,
                   GPU,
                   ALL_LAYOUT,
                   phi::WeightOnlyMatmulKernel,
                   phi::dtype::float16) {}
