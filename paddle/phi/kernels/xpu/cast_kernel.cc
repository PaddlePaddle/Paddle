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
#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out) {
  using XPUInTDType = typename XPUTypeTrait<T>::Type;
  using XPUTypeFP16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

  auto* in_data = x.data<T>();
  auto numel = x.numel();

  int r = -1;
  switch (out_dtype) {
    case phi::DataType::FLOAT32:
      r = xpu::cast<XPUInTDType, float>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<float>(out),
          numel);
      break;
    case phi::DataType::FLOAT16:
      r = xpu::cast<XPUInTDType, XPUTypeFP16>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          reinterpret_cast<XPUTypeFP16*>(
              dev_ctx.template Alloc<phi::dtype::float16>(out)),
          numel);
      break;
    case phi::DataType::INT64:
      r = xpu::cast<XPUInTDType, int64_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<int64_t>(out),
          numel);
      break;
    case phi::DataType::INT32: {
#if 0
      if (x.dims().size() == 1 && x.dims()[0] <= 4) {
        dev_ctx.Wait();
        xpu_wait();
        DenseTensor x_cpu(x.type());
        phi::Copy(dev_ctx, x, phi::CPUPlace(), true, &x_cpu);
        std::stringstream os;
        for (size_t i = 0; i < x_cpu.numel(); i++) {
          os << x_cpu.data<T>()[i] << ",";
        }
        LOG(INFO) << "cast int64->int32 tid=" << gettid() << " x_dims=" << x.dims() << " x_type=" << typeid(T).name() << " x_data=[" << os.str() << "]"; // NOLINT
      }
#endif
      r = xpu::cast<XPUInTDType, int32_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<int>(out),
          numel);
#if 0
      if (x.dims().size() == 1 && x.dims()[0] <= 4) {
        dev_ctx.Wait();
        xpu_wait();
        DenseTensor out_cpu(out->type());
        phi::Copy(dev_ctx, *out, phi::CPUPlace(), true, &out_cpu);
        std::stringstream os;
        for (size_t i = 0; i < out_cpu.numel(); i++) {
          os << out_cpu.data<int32_t>()[i] << ",";
        }
        LOG(INFO) << "cast int64->int32 tid=" << gettid() << " out_dims=" << out->dims() << " out_type=int32_t out_data=[" << os.str() << "]"; // NOLINT
      }
#endif
    } break;
    case phi::DataType::BOOL:
      r = xpu::cast<XPUInTDType, bool>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<bool>(out),
          numel);
      break;
    case phi::DataType::INT8:
      r = xpu::cast<XPUInTDType, int8_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<int8_t>(out),
          numel);
      break;
    case phi::DataType::UINT8:
      r = xpu::cast<XPUInTDType, uint8_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<uint8_t>(out),
          numel);
      break;
    case phi::DataType::FLOAT64:
      r = xpu::cast<XPUInTDType, double>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUInTDType*>(in_data),
          dev_ctx.template Alloc<double>(out),
          numel);
      break;
    default:
      PADDLE_THROW(phi::errors::Unavailable(
          "Not supported cast %d -> %d", x.dtype(), out_dtype));
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
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
                   bool,
                   int8_t,
                   uint8_t,
                   double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
