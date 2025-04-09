// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/strided_copy_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
namespace phi {

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = common::make_ddim(out_stride);
  meta.dims = common::make_ddim(dims);
  meta.offset = offset;
  out->set_meta(meta);

  PADDLE_ENFORCE_EQ(input.dims(),
                    out->dims(),
                    common::errors::InvalidArgument(
                        "Input shape(%s) must be equal with out shape(%s).",
                        input.dims(),
                        out->dims()));

  PADDLE_ENFORCE_EQ(input.numel(),
                    out->numel(),
                    common::errors::InvalidArgument(
                        "Input numel(%d) must be equal with out numel(%d).",
                        input.numel(),
                        out->numel()));

  if (input.numel() <= 0) {
    return;
  }

  PADDLE_ENFORCE_NOT_NULL(out->data<T>(),
                          common::errors::InvalidArgument(
                              "StridedCopyKernel's out tensor must complete "
                              "mutable data before call kernel."));

  // The following XPU operators have performance issues and are temporarily
  // disabled. A temporary workaround has been implemented: "First copy data to
  // CPU, perform computation using CPU operator logic, then copy results back
  // to XPU".
  /*
  // use XPUCopyTypeTrait to deal with double and int16_t copy instead of
  // XPUTypeTrait
  using XPUType = typename XPUCopyTypeTrait<T>::Type;

  int r = 0;
  auto input_data = reinterpret_cast<const XPUType*>(input.data<T>());
  auto output_data = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out));
  PADDLE_ENFORCE_NOT_NULL(output_data,
                          common::errors::InvalidArgument(
                              "StridedCopyKernel's out tensor must complete "
                              "mutable data before call kernel."));
  if (input.numel() == 1) {
    r = xpu::copy<XPUType>(dev_ctx.x_context(), input_data, output_data, 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  } else {
    r = xpu::strided_copy<XPUType>(dev_ctx.x_context(),
                                   input_data,
                                   output_data,
                                   common::vectorize<int64_t>(input.dims()),
                                   common::vectorize<int64_t>(out->dims()),
                                   common::vectorize<int64_t>(input.strides()),
                                   common::vectorize<int64_t>(out->strides()));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_copy");
  }
  */

  // wait before copy
  dev_ctx.Wait();

  // CPU buffer for input
  char* input_on_cpu = new char[input.Holder()->size()];
  memory_utils::Copy(CPUPlace(),
                     static_cast<void*>(input_on_cpu),
                     dev_ctx.GetPlace(),
                     static_cast<const void*>(input.Holder()->ptr()),
                     input.Holder()->size());

  // CPU buffer for out
  char* output_on_cpu = new char[out->Holder()->size()];
  memory_utils::Copy(CPUPlace(),
                     static_cast<void*>(output_on_cpu),
                     dev_ctx.GetPlace(),
                     static_cast<const void*>(out->Holder()->ptr()),
                     out->Holder()->size());

  // wait after copy
  dev_ctx.Wait();

  // follow paddle/phi/kernels/cpu/strided_copy_kernel.cc
  const T* input_data =
      reinterpret_cast<T*>(input_on_cpu + input.meta().offset);
  int input_rank = input.dims().size();
  const int64_t* input_dims = input.dims().Get();
  const int64_t* input_stride = input.strides().Get();

  T* output_data = reinterpret_cast<T*>(output_on_cpu + offset);
  int output_rank = meta.dims.size();
  const int64_t* output_dims = meta.dims.Get();
  const int64_t* output_stride = meta.strides.Get();

  auto numel = input.numel();

  for (int64_t i = 0; i < numel; i++) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
    for (int dim = input_rank - 1; dim >= 0; --dim) {
      input_offset += (index_tmp % input_dims[dim]) * input_stride[dim];
      index_tmp = index_tmp / input_dims[dim];
    }
    int64_t output_offset = 0;
    index_tmp = i;
    for (int dim = output_rank - 1; dim >= 0; --dim) {
      output_offset += (index_tmp % output_dims[dim]) * output_stride[dim];
      index_tmp = index_tmp / output_dims[dim];
    }
    output_data[output_offset] = input_data[input_offset];
  }

  // copy out tensor, from cpu to xpu
  memory_utils::Copy(dev_ctx.GetPlace(),
                     static_cast<void*>(out->Holder()->ptr()),
                     CPUPlace(),
                     static_cast<const void*>(output_on_cpu),
                     out->Holder()->size());
  // wait after copy
  dev_ctx.Wait();

  delete[] input_on_cpu;
  delete[] output_on_cpu;
}

}  // namespace phi

PD_REGISTER_KERNEL(strided_copy,
                   XPU,
                   ALL_LAYOUT,
                   phi::StridedCopyKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
                   ::phi::dtype::float16,
                   ::phi::dtype::bfloat16) {}
