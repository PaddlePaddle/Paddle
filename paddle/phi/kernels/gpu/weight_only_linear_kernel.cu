/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

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
                            const int32_t arch,
                            const int32_t group_size,
                            DenseTensor* out) {
#if defined(PADDLE_WITH_CUTLASS)
  PADDLE_ENFORCE_EQ(
      ((arch == 70) || (arch == 75) || (arch == 80) || (arch == 86) ||
       (arch == 89) || (arch == 90)),
      true,
      common::errors::InvalidArgument(
          "Currently, arch only support 70, 75, 80, 86, 89, 90."));
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "Please compile with cutlass to make cutlass available"));
#endif

  dev_ctx.template Alloc<T>(out);
  const T* x_data = x.data<T>();
  const int8_t* weight_data = weight.data<int8_t>();
  const T* bias_data = bias ? bias.get().data<T>() : nullptr;
  const T* weight_scale_data = weight_scale.data<T>();
  T* out_data = out->data<T>();
  const auto x_dims = x.dims();
  const auto w_dims = weight.dims();
  int n = group_size > 0 ? weight_scale.dims()[1] : weight_scale.dims()[0];
  int k = w_dims[1];
  int m = x.numel() / k;

  // m > 3: run gemm.
  if (m > 3 || (arch == 70)) {
/*
Note(Zhengzekang):
If using arch = 70, we always dispatch to weightonly Gemm,
we havenot support sm70 weightonly gemv, because sm70 weight layout is RowMajor.
*/
#if defined(PADDLE_WITH_CUTLASS)
    if (weight_dtype == "int8") {
      auto mixed_gemm_runner =
          CutlassFpAIntBGemmRunner<typename PDDataTypeTraits<T>::DataType,
                                   uint8_t>();
      int mixgemm_max_size = std::max(m, k);
      DenseTensor mixgemm_workspace;
      int64_t mixgemm_workspace_size_bytes = mixed_gemm_runner.getWorkspaceSize(
          m, mixgemm_max_size, mixgemm_max_size);
      mixgemm_workspace_size_bytes = 100 * 1024 * 1024;
      mixgemm_workspace.Resize({mixgemm_workspace_size_bytes});
      dev_ctx.template Alloc<uint8_t>(&mixgemm_workspace);
      char* mixgemm_workspace_data =
          reinterpret_cast<char*>(mixgemm_workspace.data<uint8_t>());
      if (bias_data) {
        mixed_gemm_runner.gemm_bias_act(
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                x_data),
            reinterpret_cast<const uint8_t*>(weight_data),
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                weight_scale_data),
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                bias_data),
            reinterpret_cast<typename PDDataTypeTraits<T>::DataType*>(out_data),
            m,
            n,
            k,
            group_size,
            "none",
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            dev_ctx.stream());
      } else {
        mixed_gemm_runner.gemm(
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                x_data),
            reinterpret_cast<const uint8_t*>(weight_data),
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                weight_scale_data),
            reinterpret_cast<typename PDDataTypeTraits<T>::DataType*>(out_data),
            m,
            n,
            k,
            group_size,
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
      mixgemm_workspace_size_bytes = 100 * 1024 * 1024;
      mixgemm_workspace.Resize({mixgemm_workspace_size_bytes});
      dev_ctx.template Alloc<uint8_t>(&mixgemm_workspace);
      char* mixgemm_workspace_data =
          reinterpret_cast<char*>(mixgemm_workspace.data<uint8_t>());
      if (bias_data) {
        mixed_gemm_runner.gemm_bias_act(
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                x_data),
            reinterpret_cast<const cutlass::uint4b_t*>(weight_data),
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                weight_scale_data),
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                bias_data),
            reinterpret_cast<typename PDDataTypeTraits<T>::DataType*>(out_data),
            m,
            n,
            k,
            group_size,
            "none",
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            dev_ctx.stream());
      } else {
        mixed_gemm_runner.gemm(
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                x_data),
            reinterpret_cast<const cutlass::uint4b_t*>(weight_data),
            reinterpret_cast<const typename PDDataTypeTraits<T>::DataType*>(
                weight_scale_data),
            reinterpret_cast<typename PDDataTypeTraits<T>::DataType*>(out_data),
            m,
            n,
            k,
            group_size,
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            dev_ctx.stream());
      }
    }
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "Please compile with cutlass to make cutlass available"));
#endif
  } else {  // m <= 3: gemv
    if (weight_dtype == "int8") {
      WeightOnlyGemvWrapper<T, Context>(
          dev_ctx,
          x_data,
          weight_data,
          bias_data,
          weight_scale_data,
          m,
          n,
          k,
          group_size,
          "int8",
          group_size > 0 ? "group_wise" : "per_channel",
          "None",
          out->data<T>());

    } else if (weight_dtype == "int4") {
      WeightOnlyGemvWrapper<T, Context>(
          dev_ctx,
          x_data,
          weight_data,
          bias_data,
          weight_scale_data,
          m,
          n,
          k,
          group_size,
          "int4",
          group_size > 0 ? "group_wise" : "per_channel",
          "None",
          out->data<T>());
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(weight_only_linear,
                   GPU,
                   ALL_LAYOUT,
                   phi::WeightOnlyLinearKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
