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

#include "paddle/phi/kernels/strided_copy_kernel.h"

#include <vector>

#include "paddle/common/flags.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/transpose_grad_kernel_impl.h"

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       DenseTensor* out) {
#if defined(PADDLE_WITH_CUDA)
// not support Windows
#if !defined(_WIN32)
  if (FLAGS_use_stride_kernel &&
      input.place().GetType() == phi::AllocationType::CPU &&
      out->place().GetType() == phi::AllocationType::GPU &&
      input.dtype() == out->dtype() &&
      (!input.meta().is_contiguous() || !out->meta().is_contiguous())) {
    phi::DenseTensor dst_gpu;
    if (out->meta().is_contiguous()) {
      dst_gpu = *out;
    } else {
      auto meta_dst = dst_gpu.meta();
      meta_dst.dims = out->dims();
      meta_dst.strides = meta_dst.calc_strides(out->dims());
      dst_gpu.set_meta(meta_dst);
      dev_ctx.Alloc(&dst_gpu, input.dtype());
    }

    auto src_cpu_place = input.place();
    auto dst_gpu_place = out->place();
    auto& pool = phi::DeviceContextPool::Instance();
    auto* gpu_dev_ctx = static_cast<phi::GPUContext*>(pool.Get(out->place()));
    auto stream = gpu_dev_ctx->stream();

    if (input.meta().is_contiguous()) {
      auto src_cpu_place = input.place();
      auto dst_gpu_place = out->place();
      auto size = phi::SizeOf(input.dtype()) * input.numel();
      void* dst_ptr = gpu_dev_ctx->Alloc(
          &dst_gpu,
          dst_gpu.dtype(),
          0,
          dst_gpu_place.GetType() == AllocationType::GPUPINNED);

      phi::memory_utils::Copy(
          dst_gpu_place, dst_ptr, src_cpu_place, input.data<T>(), size, stream);

    } else {
      phi::DenseTensor cpu_out;
      phi::ContiguousKernel<T, Context>(dev_ctx, input, &cpu_out);
      auto* src_ptr = cpu_out.data<T>();
      auto size = phi::SizeOf(input.dtype()) * cpu_out.numel();
      void* dst_ptr = gpu_dev_ctx->Alloc(
          &dst_gpu,
          dst_gpu.dtype(),
          0,
          dst_gpu_place.GetType() == AllocationType::GPUPINNED);

      phi::memory_utils::Copy(
          dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);
    }
    if (out != &dst_gpu) {
      PD_VISIT_ALL_TYPES(
          out->dtype(), "StridedCopyKernel", ([&] {
            phi::StridedCopyKernel<data_t, phi::GPUContext>(
                reinterpret_cast<const phi::GPUContext&>(*gpu_dev_ctx),
                dst_gpu,
                common::vectorize<int64_t>(out->dims()),
                common::vectorize<int64_t>(out->strides()),
                out->offset(),
                out);
          }));
    }

    return;
  }
#endif
#endif

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

  const T* input_data = input.data<T>();
  int input_rank = input.dims().size();
  const int64_t* input_dims = input.dims().Get();
  const int64_t* input_stride = input.strides().Get();

  T* output_data = out->data<T>();
  PADDLE_ENFORCE_NOT_NULL(output_data,
                          common::errors::InvalidArgument(
                              "StridedCopyKernel's out tensor must complete "
                              "mutable data before call kernel."));
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
}
#ifdef _WIN32
INSTANTIATE_STRIDEDCOPY_KERNEL(bool, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(uint8_t, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(uint16_t, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(uint32_t, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(uint64_t, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(int8_t, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(int16_t, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(int32_t, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(int64_t, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(float, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(double, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(dtype::float16, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(dtype::bfloat16, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(dtype::complex<float>, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(dtype::complex<double>, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(dtype::float8_e4m3fn, CPUContext)
INSTANTIATE_STRIDEDCOPY_KERNEL(dtype::float8_e5m2, CPUContext)
#endif
}  // namespace phi

PD_REGISTER_KERNEL(strided_copy,
                   CPU,
                   ALL_LAYOUT,
                   phi::StridedCopyKernel,
                   bool,
                   uint8_t,
                   uint16_t,
                   uint32_t,
                   uint64_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2) {}
