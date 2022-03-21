//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/cast_kernel.h"

#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/transform.h"

namespace phi {

template <typename InT, typename OutT>
struct CastOpTransformFunctor {
  HOSTDEVICE OutT operator()(InT in) const { return static_cast<OutT>(in); }
};

template <typename InT, typename OutT>
void CastKernelImpl(const CPUContext& dev_ctx,
                    const DenseTensor& x,
                    DenseTensor* out) {
  auto* in_begin = x.data<InT>();
  auto numel = x.numel();
  auto* in_end = in_begin + numel;

  auto* out_begin = dev_ctx.Alloc<OutT>(out);

  paddle::platform::Transform<CPUContext> trans;
  trans(dev_ctx,
        in_begin,
        in_end,
        out_begin,
        CastOpTransformFunctor<InT, OutT>());
}

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out) {
  PD_VISIT_ALL_TYPES(out_dtype, "CastKernelImpl", ([&] {
                       CastKernelImpl<T, data_t>(dev_ctx, x, out);
                     }));
}

}  // namespace phi

PD_REGISTER_KERNEL(cast,
                   CPU,
                   ALL_LAYOUT,
                   phi::CastKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   int16_t,
                   bool,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
