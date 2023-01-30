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

#include "paddle/phi/kernels/cast_kernel.h"

<<<<<<< HEAD
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
=======
#include "paddle/phi/backends/xpu/xpu_context.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

<<<<<<< HEAD
=======
// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
namespace phi {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out) {
  using XPUInTDType = typename XPUTypeTrait<T>::Type;
  using float16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

  auto* in_data = x.data<T>();
  auto numel = x.numel();

  int r = -1;
  switch (out_dtype) {
    case phi::DataType::FLOAT32:
<<<<<<< HEAD
      r = xpu::cast<XPUInTDType, float>(
=======
      r = xpu::cast_v2<XPUInTDType, float>(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<float>(out),
          numel);
      break;
    case phi::DataType::FLOAT16:
<<<<<<< HEAD
      r = xpu::cast<XPUInTDType, float16>(
=======
      r = xpu::cast_v2<XPUInTDType, float16>(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          reinterpret_cast<float16*>(
              dev_ctx.template Alloc<phi::dtype::float16>(out)),
          numel);
      break;
    case phi::DataType::INT64:
<<<<<<< HEAD
      r = xpu::cast<XPUInTDType, int64_t>(
=======
      r = xpu::cast_v2<XPUInTDType, int64_t>(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<int64_t>(out),
          numel);
      break;
    case phi::DataType::INT32:
<<<<<<< HEAD
      r = xpu::cast<XPUInTDType, int32_t>(
=======
      r = xpu::cast_v2<XPUInTDType, int32_t>(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<int>(out),
          numel);
      break;
    case phi::DataType::BOOL:
<<<<<<< HEAD
      r = xpu::cast<XPUInTDType, bool>(
=======
      r = xpu::cast_v2<XPUInTDType, bool>(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<bool>(out),
          numel);
      break;
    case phi::DataType::UINT8:
<<<<<<< HEAD
      r = xpu::cast<XPUInTDType, uint8_t>(
=======
      r = xpu::cast_v2<XPUInTDType, uint8_t>(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<uint8_t>(out),
          numel);
      break;
<<<<<<< HEAD
    case phi::DataType::FLOAT64:
      r = xpu::cast<XPUInTDType, double>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<double>(out),
          numel);
      break;
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    default:
      PADDLE_THROW(phi::errors::Unavailable(
          "Not supported cast %d -> %d", x.dtype(), out_dtype));
  }

<<<<<<< HEAD
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
=======
  PADDLE_ENFORCE_EQ(
      r,
      XPU_SUCCESS,
      phi::errors::External(
          "XPU CAST API return wrong value[%d %s].", r, XPUAPIErrorMsg[r]));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}
}  // namespace phi

PD_REGISTER_KERNEL(cast,
                   XPU,
                   ALL_LAYOUT,
                   phi::CastKernel,
                   int32_t,
                   float,
                   phi::dtype::float16,
                   int64_t,
<<<<<<< HEAD
                   bool,
                   uint8_t,
                   double) {
=======
                   bool) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
