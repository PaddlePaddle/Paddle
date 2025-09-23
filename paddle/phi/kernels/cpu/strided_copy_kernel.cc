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
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/transpose_grad_kernel_impl.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

inline int64_t divup(int64_t x, int64_t y) { return (x + y - 1) / y; }

inline void get_strides(int64_t* strides,
                        DenseTensorIterator iter,
                        int64_t ndim) {
  for (int dim = 0; dim < ndim; dim++) {
    for (int arg = 0; arg < iter.ntensors(); arg++) {
      *strides++ = iter.strides(arg)[dim];
    }
  }
  // Always at least 2d strides to support 2d for_each loops
  if (ndim < 2) {
    auto ntensors = iter.ntensors();
    std::fill_n(strides, (2 - ndim) * ntensors, 0);
  }
}

inline bool copy_transpose_valid(const DenseTensor& self,
                                 const DenseTensor& src) {
  const int MIN_SZ = 60 * 60;
  return src.numel() != 0 && src.dims().size() == 2 && src.strides()[0] == 1 &&
         src.strides()[1] == src.dims()[0] &&
         self.dims().size() == src.dims().size() && self.numel() >= MIN_SZ;
}

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       DenseTensor* out) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

  if (FLAGS_use_stride_compute_kernel &&
      input.place().GetType() == phi::AllocationType::CPU &&
      out->place().GetType() == phi::AllocationType::GPU &&
      !input.meta().is_contiguous()) {
    phi::DenseTensor dst_contig;
    phi::DenseTensor src_contig;

    if (out->meta().is_contiguous()) {
      dst_contig = *out;
    } else {
      auto meta_dst = dst_contig.meta();
      meta_dst.dims = out->dims();
      meta_dst.strides = meta_dst.calc_strides(out->dims());
      dst_contig.set_meta(meta_dst);
      dev_ctx.Alloc(&dst_contig, input.dtype());
    }

    phi::DenseTensor cpu_input = input;
    phi::DenseTensor* cpu_out = &src_contig;
    void* cpu_output_data;

    phi::DenseTensorMeta cpu_meta = cpu_input.meta();
    cpu_meta.strides = cpu_meta.calc_strides(cpu_meta.dims);
    cpu_meta.offset = 0;
    cpu_out->set_meta(cpu_meta);

    const void* cpu_input_data = cpu_input.data();
    cpu_output_data = malloc(phi::SizeOf(cpu_input.dtype()) * cpu_out->numel());

    if (copy_transpose_valid(*cpu_out, cpu_input)) {
      int64_t BLOCK_SZ = 60;
      void* buf = malloc(phi::SizeOf(input.dtype()) * BLOCK_SZ * BLOCK_SZ);

      const T* sp = reinterpret_cast<const T*>(cpu_input_data);
      T* rp = reinterpret_cast<T*>(cpu_output_data);
      T* bp = reinterpret_cast<T*>(buf);

      int64_t NR = cpu_out->dims()[0];
      int64_t NC = cpu_out->dims()[1];

      for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
        for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
          const T* spo = sp + R + C * NR;
          T* rpo = rp + C + R * NC;

          int nr = std::min(NR - R, BLOCK_SZ);
          int nc = std::min(NC - C, BLOCK_SZ);

          // 1. copy columns from src to buf
          for (int c = 0; c < nc; c++) {
            memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(T));
          }

          // 2. transpose buf in place
          int rc_max = std::max(nr, nc);
          int rc_min = std::min(nr, nc);
          for (int r = 0; r < rc_max; r++) {
            int end = std::min(r, rc_min);
            for (int c = 0; c < end; c++) {
              T tmp = bp[r + BLOCK_SZ * c];
              bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
              bp[r * BLOCK_SZ + c] = tmp;
            }
          }

          // 3. copy rows from buf to dst
          for (int r = 0; r < nr; r++) {
            memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(T));
          }
        }
      }
      free(buf);

    } else {
#if defined(_OPENMP)
      phi::DenseTensorIteratorConfig config;
      config.add_output(*cpu_out);
      config.add_const_input(cpu_input);
      config.is_alloc_out_ = true;
      phi::DenseTensorIterator iter = config.build();

      std::vector<int64_t> tmp_strides(
          iter.ntensors() * static_cast<size_t>(std::max(iter.ndim(), 2)));

      get_strides(tmp_strides.data(), iter, iter.ndim());

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
          num_threads = std::min(num_threads, divup((end - begin), grain_size));
        }

        int64_t tid = omp_get_thread_num();
        int64_t chunk_size = divup((end - begin), num_threads);
        int64_t begin_tid = begin + tid * chunk_size;

        if (begin_tid < end) {
          int64_t range_start = begin_tid;
          int64_t range_end = std::min(end, chunk_size + begin_tid);

          Range range(range_start, range_end);
          auto counter = DimCounter(iter.shape(), range);
          while (!counter.is_done()) {
            const auto v_ndim = counter.values.size();
            const char* tmp_in_data = in_ptr;
            char* tmp_out_data = out_ptr;
            for (size_t dim = 0; dim < v_ndim; dim++) {
              int64_t value = counter.values[dim];
              tmp_out_data += value * whole_stride[dim * iter.ntensors() + 0];
              tmp_in_data += value * whole_stride[dim * iter.ntensors() + 1];
            }

            auto step = counter.max_2d_step();

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

            counter.increment(step);
          }
        }
      }
#else
      phi::ContiguousKernel<T, Context>(dev_ctx, input, cpu_out);
#endif
    }

    auto src_cpu_place = input.place();
    auto dst_gpu_place = out->place();
    auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();

#if defined(_OPENMP)

    auto* src_ptr = cpu_output_data;

#else
    auto* src_ptr = cpu_out->data<T>();
#endif

    auto size = phi::SizeOf(input.dtype()) * src_contig.numel();
    void* dst_ptr =
        dev_ctx.Alloc(&dst_contig,
                      dst_contig.dtype(),
                      0,
                      dst_gpu_place.GetType() == AllocationType::GPUPINNED);

    phi::memory_utils::Copy(
        dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);

    free(cpu_output_data);

    if (out != &dst_contig) {
      PD_VISIT_ALL_TYPES(
          out->dtype(), "StridedCopyKernel", ([&] {
            phi::StridedCopyKernel<data_t, phi::GPUContext>(
                reinterpret_cast<const phi::GPUContext&>(dev_ctx),
                dst_contig,
                common::vectorize<int64_t>(out->dims()),
                common::vectorize<int64_t>(out->strides()),
                out->offset(),
                out);
          }));
    }

    return;
  }
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
                   ::phi::float16,
                   ::phi::bfloat16,
                   ::phi::complex64,
                   ::phi::complex128,
                   ::phi::float8_e4m3fn,
                   ::phi::float8_e5m2) {}
