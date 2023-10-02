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
#include "paddle/phi/kernels/weight_only_linear_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/weight_only_gemv.h"
#if defined(PADDLE_WITH_CUTLASS)
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#endif

namespace phi {

template <typename T, typename Context>
void WeightOnlyLinearKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            const paddle::optional<DenseTensor>& bias,
                            const DenseTensor& weight_scale,
                            const std::string& weight_dtype,
                            DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const T* x_data = x.data<T>();
  const int8_t* weight_data = weight.data<int8_t>();
  const T* bias_data = bias ? bias.get().data<T>() : nullptr;
  const float* weight_scale_data = weight_scale.data<float>();
  T* out_data = out->data<T>();
  const auto x_dims = x.dims();
  const auto w_dims = weight.dims();
  int n = weight_scale.dims()[0];
  int k = w_dims[1];
  int m = x.numel() / k;

  // m > 1: run gemm
  if (m > 1 || weight_dtype == "int4") {
#if defined(PADDLE_WITH_CUTLASS)
    if (weight_dtype == "int8") {
      auto mixed_gemm_runner =
          CutlassFpAIntBGemmRunner<typename PDDataTypeTraits<T>::DataType,
                                   uint8_t>();
      int mixgemm_max_size = std::max(m, k);
      DenseTensor mixgemm_workspace;
      int64_t mixgemm_workspace_size_bytes = mixed_gemm_runner.getWorkspaceSize(
          m, mixgemm_max_size, mixgemm_max_size);

      mixgemm_workspace.Resize({mixgemm_workspace_size_bytes});
      dev_ctx.template Alloc<uint8_t>(&mixgemm_workspace);
      char* mixgemm_workspace_data =
          reinterpret_cast<char*>(mixgemm_workspace.data<uint8_t>());
      if (bias_data) {
        mixed_gemm_runner.gemm_bias_act(
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                x_data),
            reinterpret_cast<const uint8_t*>(weight_data),
            weight_scale_data,
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                bias_data),
            reinterpret_cast<typename PDDataTypeTraits<T>::DataType*>(out_data),
            m,
            n,
            k,
            "none",
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            dev_ctx.stream());
      } else {
        mixed_gemm_runner.gemm(
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                x_data),
            reinterpret_cast<const uint8_t*>(weight_data),
            weight_scale_data,
            reinterpret_cast<typename PDDataTypeTraits<T>::DataType*>(out_data),
            m,
            n,
            k,
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            dev_ctx.stream());
      }
    } else {
      auto mixed_gemm_runner =
          CutlassFpAIntBGemmRunner<typename PDDataTypeTraits<T>::DataType,
                                   cutlass::uint4b_t>();
      int mixgemm_max_size = std::max(m, k);
      DenseTensor mixgemm_workspace;
      int64_t mixgemm_workspace_size_bytes = mixed_gemm_runner.getWorkspaceSize(
          m, mixgemm_max_size, mixgemm_max_size);

      mixgemm_workspace.Resize({mixgemm_workspace_size_bytes});
      dev_ctx.template Alloc<uint8_t>(&mixgemm_workspace);
      char* mixgemm_workspace_data =
          reinterpret_cast<char*>(mixgemm_workspace.data<uint8_t>());
      if (bias_data) {
        mixed_gemm_runner.gemm_bias_act(
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                x_data),
            reinterpret_cast<const cutlass::uint4b_t*>(weight_data),
            weight_scale_data,
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                bias_data),
            reinterpret_cast<typename PDDataTypeTraits<T>::DataType*>(out_data),
            m,
            n,
            k,
            "none",
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            dev_ctx.stream());
      } else {
        mixed_gemm_runner.gemm(
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                x_data),
            reinterpret_cast<const cutlass::uint4b_t*>(weight_data),
            weight_scale_data,
            reinterpret_cast<typename PDDataTypeTraits<T>::DataType*>(out_data),
            m,
            n,
            k,
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            dev_ctx.stream());
      }
    }
#else
    PADDLE_THROW(phi::errors::Unimplemented(
        "Please compile with cutlass to make cutlass available"));
#endif
  } else {  // m == 1: gemv
    if (weight_dtype == "int8") {
      GemvWeightonlyInt8Wrapper<T, Context>(dev_ctx,
                                            x_data,
                                            weight_data,
                                            bias_data,
                                            weight_scale_data,
                                            n,
                                            k,
                                            "None",
                                            out->data<T>());
    }  // TODO(lizhenyun) support weight_only_gemv for int4.
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(weight_only_linear,
                   GPU,
                   ALL_LAYOUT,
                   phi::WeightOnlyLinearKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
