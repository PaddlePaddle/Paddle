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
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/transpose_grad_kernel_impl.h"

#if defined(PADDLE_WITH_OPENMP)
#include <omp.h>
#else
#include "paddle/phi/kernels/contiguous_kernel.h"
#endif

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {
inline int64_t DivUp(const int64_t& x, const int64_t& y) {
  return (x + y - 1) / y;
}

inline void DealWithStride(const DenseTensorIterator& iter, int64_t* strides) {
  for (int dim = 0; dim < iter.ndim(); dim++) {
    for (int arg = 0; arg < iter.ntensors(); arg++) {
      *strides++ = iter.strides(arg)[dim];
    }
  }
  if (iter.ndim() < 2) {
    std::fill_n(strides, (2 - iter.ndim()) * iter.ntensors(), 0);
  }
}

inline bool FastTransposeCopyValid(const DenseTensor& self,
                                   const DenseTensor& src) {
  constexpr int64_t MIN_NUMEL = 360;
  return src.numel() != 0 && src.dims().size() == 2 && src.strides()[0] == 1 &&
         src.strides()[1] == src.dims()[0] &&
         self.dims().size() == src.dims().size() && self.numel() >= MIN_NUMEL;
}

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
  if (FLAGS_use_stride_kernel && FLAGS_use_stride_compute_kernel &&
      input.place().GetType() == phi::AllocationType::CPU &&
      out->place().GetType() == phi::AllocationType::GPU &&
      input.dtype() == out->dtype() && !input.meta().is_contiguous()) {
    phi::DenseTensor dst_gpu;
    phi::DenseTensor src_cpu;

    if (out->meta().is_contiguous()) {
      dst_gpu = *out;
    } else {
      auto meta_dst = dst_gpu.meta();
      meta_dst.dims = out->dims();
      meta_dst.strides = meta_dst.calc_strides(out->dims());
      dst_gpu.set_meta(meta_dst);
      dev_ctx.Alloc(&dst_gpu, input.dtype());
    }

    phi::DenseTensor cpu_input = input;
    phi::DenseTensor* cpu_out = &src_cpu;
    void* cpu_output_data;

    phi::DenseTensorMeta cpu_meta = cpu_input.meta();
    cpu_meta.strides = cpu_meta.calc_strides(cpu_meta.dims);
    cpu_meta.offset = 0;
    cpu_out->set_meta(cpu_meta);

#if defined(PADDLE_WITH_OPENMP)
    dev_ctx.HostAlloc(cpu_out, cpu_out->dtype());
#endif
    const void* cpu_input_data = cpu_input.data();
    cpu_output_data = malloc(phi::SizeOf(cpu_input.dtype()) * cpu_out->numel());

    if (FastTransposeCopyValid(*cpu_out, cpu_input)) {
      constexpr int64_t TRANS_NUMEL = 60;
      void* trans_buffer =
          malloc(phi::SizeOf(input.dtype()) * TRANS_NUMEL * TRANS_NUMEL);

      const T* tmp_src_ptr = reinterpret_cast<const T*>(cpu_input_data);
#if defined(PADDLE_WITH_OPENMP)
      T* tmp_out_ptr = reinterpret_cast<T*>(cpu_output_data);
#else
      T* tmp_out_ptr = cpu_out->data<T>();
#endif
      T* tmp_buf_ptr = reinterpret_cast<T*>(trans_buffer);

      int64_t dim0 = cpu_out->dims()[0];
      int64_t dim1 = cpu_out->dims()[1];

      for (int64_t d0 = 0; d0 < dim0; d0 += TRANS_NUMEL) {
        for (int64_t d1 = 0; d1 < dim1; d1 += TRANS_NUMEL) {
          const T* src_ptr_inter = tmp_src_ptr + d0 + d1 * dim0;
          T* out_ptr_inter = tmp_out_ptr + d1 + d0 * dim1;

          int nr = std::min(dim0 - d0, TRANS_NUMEL);
          int nc = std::min(dim1 - d1, TRANS_NUMEL);

          for (int c = 0; c < nc; c++) {
            memcpy(tmp_buf_ptr + c * TRANS_NUMEL,
                   src_ptr_inter + c * dim0,
                   nr * sizeof(T));
          }

          int rc_max = std::max(nr, nc);
          int rc_min = std::min(nr, nc);
          for (int r = 0; r < rc_max; r++) {
            int end = std::min(r, rc_min);
            for (int c = 0; c < end; c++) {
              T tmp = tmp_buf_ptr[r + TRANS_NUMEL * c];
              tmp_buf_ptr[r + TRANS_NUMEL * c] =
                  tmp_buf_ptr[r * TRANS_NUMEL + c];
              tmp_buf_ptr[r * TRANS_NUMEL + c] = tmp;
            }
          }

          for (int r = 0; r < nr; r++) {
            memcpy(out_ptr_inter + r * dim1,
                   tmp_buf_ptr + r * TRANS_NUMEL,
                   nc * sizeof(T));
          }
        }
      }
      free(trans_buffer);
    } else {
#if defined(PADDLE_WITH_OPENMP)
      phi::DenseTensorIteratorConfig config;
      config.add_output(*cpu_out);
      config.add_const_input(cpu_input);
      config.is_alloc_out_ = true;
      phi::DenseTensorIterator iter = config.build();

      std::vector<int64_t> tmp_strides(
          iter.ntensors() * static_cast<size_t>(std::max(iter.ndim(), 2)));

      DealWithStride(iter, tmp_strides.data());

      std::vector<int64_t> out_stride(tmp_strides.begin() + iter.ntensors(),
                                      tmp_strides.end());

      std::vector<int64_t> output_stride = iter.strides(0);
      std::vector<int64_t> input_stride = iter.strides(1);

      const int64_t& numel = iter.numel();

      const char* in_ptr = reinterpret_cast<const char*>(cpu_input_data);
      char* out_ptr = reinterpret_cast<char*>(cpu_output_data);

      int64_t end = numel;
      int64_t begin = 0;
      int64_t grain_size = 32768;

      int64_t* whole_stride = tmp_strides.data();

      omp_set_num_threads(std::thread::hardware_concurrency());

#pragma omp parallel
      {
        int64_t num_threads = omp_get_num_threads();

        if (grain_size > 0) {
          num_threads = std::min(num_threads, DivUp((end - begin), grain_size));
        }

        int64_t tid = omp_get_thread_num();
        int64_t chunk_size = DivUp((end - begin), num_threads);
        int64_t begin_tid = begin + tid * chunk_size;

        if (begin_tid < end) {
          int64_t range_start = begin_tid;
          int64_t range_end = std::min(end, chunk_size + begin_tid);

          auto dimiter = DimIter(iter.shape(), range_start, range_end);
          while (!dimiter.iter_to_end()) {
            const auto v_ndim = dimiter.values.size();
            const char* tmp_in_data = in_ptr;
            char* tmp_out_data = out_ptr;
            for (size_t dim = 0; dim < v_ndim; dim++) {
              int64_t value = dimiter.values[dim];
              tmp_out_data += value * whole_stride[dim * iter.ntensors() + 0];
              tmp_in_data += value * whole_stride[dim * iter.ntensors() + 1];
            }

            auto step = dimiter.iter_for_step();

            for (int64_t i = 0; i < step[1]; i++) {
              for (int64_t j = 0; j < step[0]; j++) {
                const char* real_in_ptr = tmp_in_data + j * whole_stride[1];
                char* real_out_ptr = tmp_out_data + j * whole_stride[0];

                *reinterpret_cast<T*>(real_out_ptr) =
                    *reinterpret_cast<const T*>(real_in_ptr);
              }
              tmp_in_data = tmp_in_data + out_stride[1];
              tmp_out_data = tmp_out_data + out_stride[0];
            }

            dimiter.iter_to_next(step);
          }
        }
      }
#else
      phi::ContiguousKernel<T, Context>(dev_ctx, input, cpu_out);
#endif
    }

    auto src_cpu_place = input.place();
    auto dst_gpu_place = out->place();

    auto& pool = phi::DeviceContextPool::Instance();
    auto* gpu_dev_ctx = static_cast<phi::GPUContext*>(pool.Get(out->place()));
    auto stream = gpu_dev_ctx->stream();
#if defined(PADDLE_WITH_OPENMP)
    auto* src_ptr = cpu_output_data;
#else
    auto* src_ptr = cpu_out->data<T>();
#endif

    auto size = phi::SizeOf(input.dtype()) * src_cpu.numel();
    void* dst_ptr = gpu_dev_ctx->Alloc(
        &dst_gpu,
        dst_gpu.dtype(),
        0,
        dst_gpu_place.GetType() == AllocationType::GPUPINNED);

    phi::memory_utils::Copy(
        dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);

    free(cpu_output_data);
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
