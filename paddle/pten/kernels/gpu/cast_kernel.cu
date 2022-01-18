// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/kernels/cast_kernel.h"

#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/funcs/elementwise_base.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/device/gpu/gpu_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/float16.h"

namespace pten {

template <typename InT, typename OutT>
struct CastFuctor {
  __device__ __forceinline__ OutT operator()(const InT x) const {
    return static_cast<OutT>(x);
  }
};

template <typename InT, typename OutT>
void CastCUDAKernelImpl(const GPUContext& dev_ctx,
                        const DenseTensor& x,
                        DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  std::vector<DenseTensor*> outputs;
  inputs.emplace_back(&x);
  outputs.emplace_back(out);
  out->mutable_data<OutT>();
  pten::funcs::LaunchSameDimsElementwiseCudaKernel<ElementwiseType::kUnary,
                                                   InT,
                                                   OutT>(
      dev_ctx, inputs, &outputs, CastFuctor<InT, OutT>());
}

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out) {
  PD_VISIT_ALL_TYPES(out_dtype, "CastCUDAKernelImpl", ([&] {
                       CastCUDAKernelImpl<T, data_t>(dev_ctx, x, out);
                     }));
}

}  // namespace pten

#define PTEN_REGISTER_CAST_CUDA_BASE_TYPE(op_name, ...) \
  PT_REGISTER_KERNEL(cast,                              \
                     GPU,                               \
                     ALL_LAYOUT,                        \
                     pten::CastKernel,                  \
                     float,                             \
                     double,                            \
                     int,                               \
                     int64_t,                           \
                     int16_t,                           \
                     bool,                              \
                     uint8_t,                           \
                     paddle::platform::float16,         \
                     paddle::platform::complex<float>,  \
                     paddle::platform::complex<double>, \
                     ##__VA_ARGS__) {                   \
    kernel->OutputAt(0).SetDataType(                    \
        paddle::experimental::DataType::UNDEFINED);     \
  }

#if !defined(PADDLE_WITH_HIP)
PTEN_REGISTER_CAST_CUDA_BASE_TYPE(cast, paddle::platform::bfloat16)
#else
PTEN_REGISTER_CAST_CUDA_BASE_TYPE(cast)
#endif
