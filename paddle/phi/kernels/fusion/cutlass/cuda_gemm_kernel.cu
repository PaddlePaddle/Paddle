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

#include <glog/logging.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/cutlass/cuda_gemm/cuda_gemm.h"
#include "paddle/phi/backends/dynload/cutlass_cuda_gemm.h"


namespace phi {
namespace fusion {
namespace cutlass_internal {

typedef bool (*func)(phi::fusion::cutlass_internal::GemmParams);

template <typename T, typename Context>
void CudaGemm(const Context& ctx,
                             const DenseTensor& input,
                             const DenseTensor& w,
                             DenseTensor* output) {
  // printf("zhangsishuai\n");
  ctx.template Alloc<int32_t>(output);
  auto input_dims = input.dims();
  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      2UL,
      common::errors::InvalidArgument(
          "The input tensor dimensions should be 2, but got %d.",
          input_dims.size()));
  auto weight_dims = w.dims();
  PADDLE_ENFORCE_EQ(
      weight_dims.size(),
      2UL,
      common::errors::InvalidArgument(
          "The w tensor dimensions should be 2, but got %d.",
          input_dims.size()));
  printf("zhangsishuai %d %d %d %d\n", input_dims[0], input_dims[1], weight_dims[0], weight_dims[1]);
  // PADDLE_ENFORCE_EQ(weight_dims.size(),
  //                   2UL,
  //                   common::errors::InvalidArgument(
  //                       "In gemm_epilogue kernel, weight_dims should be 2."));
  auto out_dims = output->dims();
  // printf("zhangsishuai   222\n");
  
  // const int batch = input_dims[0];
  const int m = input_dims[0];
  const int n = input_dims[1];
  
  PADDLE_ENFORCE_EQ(
      input_dims[1],
      weight_dims[0],
      common::errors::InvalidArgument(
          "The w tensor dimensions should be 2, but got %d.",
          input_dims.size()));
  const int k = weight_dims[1];

  auto get_phi_dtype = [&](decltype(input.dtype()) x_type)
      -> int {
    switch (x_type) {
      case phi::DataType::INT8:
        return 4;
      case phi::DataType::FLOAT8_E5M2:
        return 5;
    }
  };

  GemmParams params = {
      reinterpret_cast<const void*>(input.data<T>()),
      reinterpret_cast<const void*>(w.data<T>()),
      reinterpret_cast<void*>(output->data<int32_t>()),
      m,
      n,
      k,
      get_phi_dtype(input.dtype()),
      get_phi_dtype(input.dtype()) == 4 ? 5 : 1,
      ctx.stream(),
  };

  void* dlhandler = phi::dynload::GetCutlassCudaGemmHandle();
  if (dlhandler == NULL) {

    PADDLE_THROW(common::errors::Fatal("no cudaGemmDispatcher dlhandler cutlass kernel "));
  }
  PADDLE_ENFORCE_NOT_NULL(
      dlhandler,
      common::errors::NotFound("Fail to get CutlassCudaGemm handler."));

  func cuda_gemm_func = NULL;
  cuda_gemm_func = (func)(dlsym(dlhandler, "cudaGemmDispatcher"));
  if(cuda_gemm_func == NULL){
    PADDLE_THROW(common::errors::Fatal("no cudaGemmDispatcher cutlass kernel "));
  }
  if (!cuda_gemm_func(params)) {
    PADDLE_THROW(
        common::errors::Fatal("no cuda_core_gemm cutlass kernel "));
  }
}
}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(cuda_gemm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::CudaGemm,
                   int8_t,
                   phi::dtype::float8_e4m3fn) {}
