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

#pragma once

#include "paddle/phi/api/include/tensor.h"

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"

#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"

#include "paddle/phi/kernels/funcs/index_elementwise.h"

#if defined(_OPENMP)
#include <omp.h>

namespace phi {

inline int64_t divup(int64_t x, int64_t y) { return (x + y - 1) / y; }

// Note: This function may not always provide the optimal solution.
// It performs better when running on a system with a powerful CPU;
// otherwise, it may cause performance degradation.
//
// If performance degradation occurs, you can modify the Python code as follows:
//
// Case: x is non-contiguous and located on the CPU,
//       while y is on the GPU
//
// Change:
//     y.copy_(x)
// To:
//     x = x.cuda(0)
//     y.copy_(x)
//
// This adjustment can improve performance.

// TODO(wangjinheng): CPU vectorization

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

bool copy_transpose_valid(const DenseTensor& self, const DenseTensor& src) {
  const int MIN_SZ = 60 * 60;
  return src.numel() != 0 && src.dims().size() == 2 && src.strides()[0] == 1 &&
         src.strides()[1] == src.dims()[0] &&
         self.dims().size() == src.dims().size() && self.numel() >= MIN_SZ;
}

template <typename Context>
void FastCPUCopy(const Context& dev_ctx,
                 const DenseTensor& src_tensor,
                 const phi::Place& target_place,
                 DenseTensor* dst_tensor) {
  void* output_data;

  phi::DenseTensor dst_contig;
  phi::DenseTensor src_contig;

  if (dst_tensor->meta().is_contiguous()) {
    dst_contig = *dst_tensor;
  } else {
    auto meta_dst = dst_contig.meta();
    meta_dst.dims = dst_tensor->dims();
    meta_dst.strides = meta_dst.calc_strides(dst_tensor->dims());
    dst_contig.set_meta(meta_dst);
    dev_ctx.Alloc(&dst_contig, src_tensor.dtype());
  }

  phi::DenseTensor input = src_tensor;
  phi::DenseTensor* out = &src_contig;

  phi::DenseTensorMeta meta = input.meta();
  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);

  const void* input_data = input.data();
  output_data = malloc(phi::SizeOf(input.dtype()) * out->numel());

  if (copy_transpose_valid(*out, input)) {
    int64_t BLOCK_SZ = 60;
    void* buf = malloc(phi::SizeOf(input.dtype()) * BLOCK_SZ * BLOCK_SZ);

    if (phi::SizeOf(input.dtype()) == 4) {
      const int32_t* sp = reinterpret_cast<const int32_t*>(input_data);
      int32_t* rp = reinterpret_cast<int32_t*>(output_data);
      int32_t* bp = reinterpret_cast<int32_t*>(buf);

      DenseTensor src = *out;

      int64_t NR = src.dims()[0];
      int64_t NC = src.dims()[1];

      for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
        for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
          const int32_t* spo = sp + R + C * NR;
          int32_t* rpo = rp + C + R * NC;

          int nr = std::min(NR - R, BLOCK_SZ);
          int nc = std::min(NC - C, BLOCK_SZ);

          // 1. copy columns from src to buf
          for (int c = 0; c < nc; c++) {
            memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(int32_t));
          }

          // 2. transpose buf in place
          int rc_max = std::max(nr, nc);
          int rc_min = std::min(nr, nc);
          for (int r = 0; r < rc_max; r++) {
            int end = std::min(r, rc_min);
            for (int c = 0; c < end; c++) {
              int32_t tmp = bp[r + BLOCK_SZ * c];
              bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
              bp[r * BLOCK_SZ + c] = tmp;
            }
          }

          // 3. copy rows from buf to dst
          for (int r = 0; r < nr; r++) {
            memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(int32_t));
          }
        }
      }

    } else if (phi::SizeOf(input.dtype()) == 2) {
      const int16_t* sp = reinterpret_cast<const int16_t*>(input_data);
      int16_t* rp = reinterpret_cast<int16_t*>(output_data);
      int16_t* bp = reinterpret_cast<int16_t*>(buf);

      DenseTensor src = *out;

      int64_t NR = src.dims()[0];
      int64_t NC = src.dims()[1];

      for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
        for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
          const int16_t* spo = sp + R + C * NR;
          int16_t* rpo = rp + C + R * NC;

          int nr = std::min(NR - R, BLOCK_SZ);
          int nc = std::min(NC - C, BLOCK_SZ);

          // 1. copy columns from src to buf
          for (int c = 0; c < nc; c++) {
            memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(int16_t));
          }

          // 2. transpose buf in place
          int rc_max = std::max(nr, nc);
          int rc_min = std::min(nr, nc);
          for (int r = 0; r < rc_max; r++) {
            int end = std::min(r, rc_min);
            for (int c = 0; c < end; c++) {
              int16_t tmp = bp[r + BLOCK_SZ * c];
              bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
              bp[r * BLOCK_SZ + c] = tmp;
            }
          }

          // 3. copy rows from buf to dst
          for (int r = 0; r < nr; r++) {
            memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(int16_t));
          }
        }
      }

    } else if (phi::SizeOf(input.dtype()) == 1) {
      const int8_t* sp = reinterpret_cast<const int8_t*>(input_data);
      int8_t* rp = reinterpret_cast<int8_t*>(output_data);
      int8_t* bp = reinterpret_cast<int8_t*>(buf);

      DenseTensor src = *out;

      int64_t NR = src.dims()[0];
      int64_t NC = src.dims()[1];

      for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
        for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
          const int8_t* spo = sp + R + C * NR;
          int8_t* rpo = rp + C + R * NC;

          int nr = std::min(NR - R, BLOCK_SZ);
          int nc = std::min(NC - C, BLOCK_SZ);

          // 1. copy columns from src to buf
          for (int c = 0; c < nc; c++) {
            memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(int8_t));
          }

          // 2. transpose buf in place
          int rc_max = std::max(nr, nc);
          int rc_min = std::min(nr, nc);
          for (int r = 0; r < rc_max; r++) {
            int end = std::min(r, rc_min);
            for (int c = 0; c < end; c++) {
              int8_t tmp = bp[r + BLOCK_SZ * c];
              bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
              bp[r * BLOCK_SZ + c] = tmp;
            }
          }

          // 3. copy rows from buf to dst
          for (int r = 0; r < nr; r++) {
            memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(int8_t));
          }
        }
      }

    } else {
      PADDLE_THROW(
          ::common::errors::InvalidArgument("Copy Dtype not Implemented"));
    }

    free(buf);

  } else {
    phi::DenseTensorIteratorConfig config;
    config.add_output(*out);
    config.add_const_input(input);
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

    int all_num_threads = 96;

    const char* env_threads = std::getenv("FLAGS_num_threads");
    if (env_threads != nullptr) {
      int parsed_threads = std::atoi(env_threads);
      if (parsed_threads > 0) {
        all_num_threads = parsed_threads;
      }
    }

    omp_set_num_threads(all_num_threads);

    const char* in_ptr = reinterpret_cast<const char*>(input_data);
    char* out_ptr = reinterpret_cast<char*>(output_data);

    int64_t end = numel;
    int64_t begin = 0;
    int64_t grain_size = 32768;

    int64_t* whole_stride = tmp_strides.data();
    int64_t* load_stride = &(whole_stride[1]);

    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;

    if (phi::SizeOf(input.dtype()) == 4) {
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
            for (int dim = 0; dim < v_ndim; dim++) {
              int64_t value = counter.values[dim];
              tmp_out_data += value * whole_stride[dim * iter.ntensors() + 0];
              tmp_in_data += value * whole_stride[dim * iter.ntensors() + 1];
            }

            auto step = counter.max_2d_step();

            for (int64_t i = 0; i < step[1]; i++) {
              for (int64_t j = 0; j < step[0]; j++) {
                const char* real_in_ptr = tmp_in_data + j * whole_stride[1];
                char* real_out_ptr = tmp_out_data + j * whole_stride[0];

                *reinterpret_cast<int32_t*>(real_out_ptr) =
                    *reinterpret_cast<const int32_t*>(real_in_ptr);
              }
              tmp_in_data = tmp_in_data + out_stride[1];
              tmp_out_data = tmp_out_data + out_stride[0];
            }

            counter.increment(step);
          }
        }
      }

    } else if (phi::SizeOf(input.dtype()) == 2) {
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
            for (int dim = 0; dim < v_ndim; dim++) {
              int64_t value = counter.values[dim];
              tmp_out_data += value * whole_stride[dim * iter.ntensors() + 0];
              tmp_in_data += value * whole_stride[dim * iter.ntensors() + 1];
            }

            auto step = counter.max_2d_step();

            for (int64_t i = 0; i < step[1]; i++) {
              for (int64_t j = 0; j < step[0]; j++) {
                const char* real_in_ptr = tmp_in_data + j * whole_stride[1];
                char* real_out_ptr = tmp_out_data + j * whole_stride[0];

                *reinterpret_cast<int16_t*>(real_out_ptr) =
                    *reinterpret_cast<const int16_t*>(real_in_ptr);
              }
              tmp_in_data = tmp_in_data + out_stride[1];
              tmp_out_data = tmp_out_data + out_stride[0];
            }

            counter.increment(step);
          }
        }
      }

    } else if (phi::SizeOf(input.dtype()) == 1) {
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
            for (int dim = 0; dim < v_ndim; dim++) {
              int64_t value = counter.values[dim];
              tmp_out_data += value * whole_stride[dim * iter.ntensors() + 0];
              tmp_in_data += value * whole_stride[dim * iter.ntensors() + 1];
            }

            auto step = counter.max_2d_step();

            for (int64_t i = 0; i < step[1]; i++) {
              for (int64_t j = 0; j < step[0]; j++) {
                const char* real_in_ptr = tmp_in_data + j * whole_stride[1];
                char* real_out_ptr = tmp_out_data + j * whole_stride[0];

                *reinterpret_cast<int8_t*>(real_out_ptr) =
                    *reinterpret_cast<const int8_t*>(real_in_ptr);
              }
              tmp_in_data = tmp_in_data + out_stride[1];
              tmp_out_data = tmp_out_data + out_stride[0];
            }

            counter.increment(step);
          }
        }
      }

    } else {
      PADDLE_THROW(
          ::common::errors::InvalidArgument("Copy Dtype not Implemented"));
    }
  }

  auto src_cpu_place = src_tensor.place();
  auto dst_gpu_place = target_place;
  auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();

  auto* src_ptr = output_data;

  auto size = phi::SizeOf(src_tensor.dtype()) * src_contig.numel();
  void* dst_ptr =
      dev_ctx.Alloc(&dst_contig,
                    dst_contig.dtype(),
                    0,
                    target_place.GetType() == AllocationType::GPUPINNED);

  phi::memory_utils::Copy(
      dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);

  free(output_data);

  if (dst_tensor != &dst_contig) {
    PD_VISIT_ALL_TYPES(dst_tensor->dtype(), "StridedCopyKernel", ([&] {
                         phi::StridedCopyKernel<data_t, phi::GPUContext>(
                             reinterpret_cast<const phi::GPUContext&>(dev_ctx),
                             dst_contig,
                             common::vectorize<int64_t>(dst_tensor->dims()),
                             common::vectorize<int64_t>(dst_tensor->strides()),
                             dst_tensor->offset(),
                             dst_tensor);
                       }));
  }

  return;
}

}  // namespace phi

#endif
