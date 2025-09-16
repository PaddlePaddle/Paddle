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

  phi::DenseTensorIteratorConfig config;
  config.add_output(*out);
  config.add_const_input(input);
  phi::DenseTensorIterator iter = config.build();

  std::vector<int64_t> output_stride = iter.strides(0);
  std::vector<int64_t> input_stride = iter.strides(1);

  const int64_t& numel = iter.numel();

  omp_set_num_threads(std::thread::hardware_concurrency());

  Range range(0, numel);
  auto counter = DimCounter(iter.shape(), range);

  const char* in_ptr = reinterpret_cast<const char*>(input_data);
  char* out_ptr = reinterpret_cast<char*>(output_data);

  while (!counter.is_done()) {
    auto step = counter.max_2d_step();
    int step_all = step[0] * step[1];

    int64_t end = step_all;
    int64_t begin = 0;
    int64_t grain_size = 32768;

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
        for (int64_t idx = begin_tid; idx < chunk_size + begin_tid; idx++) {
          if (idx >= end) break;
          int outer_i = idx / step[1];
          int inner_i = idx % step[1];
          int base_offset = outer_i * iter.strides(1)[0];
          int input_offset = base_offset + inner_i * iter.strides(1)[1];
          int output_offset =
              (outer_i * step[1] + inner_i) * iter.strides(1)[0];

          char* const out_data = out_ptr + output_offset;
          const char* const in_data = in_ptr + input_offset;

          *reinterpret_cast<int32_t*>(out_data) =
              *reinterpret_cast<const int32_t*>(in_data);
        }
      }
    }

    counter.increment(step);
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
