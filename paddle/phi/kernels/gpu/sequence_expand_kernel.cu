// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/impl/sequence_expand_kernel_impl.h"

namespace phi {

template <typename T>
static inline int ExpandByMemoryCopy(const phi::GPUContext& context,
                                     const DenseTensor& x,
                                     DenseTensor* out,
                                     const phi::Vector<size_t>& x_lod,
                                     const phi::Vector<size_t>& ref_lod,
                                     bool do_copy) {
  auto out_data = out->data<T>();
  auto x_data = x.data<T>();

  const auto& gpu_place = context.GetPlace();

  int x_item_length = x.numel() / x.dims()[0];
  int out_offset = 0;
  int num_copies = 0;
  for (size_t i = 1; i < ref_lod.size(); ++i) {
    int repeat_num = ref_lod[i] - ref_lod[i - 1];
    int x_start = x_lod[i - 1];
    int x_end = x_lod[i];
    int x_seq_len = x_end - x_start;
    if (repeat_num > 0) {
      if (do_copy) {
        int out_start = out_offset;
        if (out->lod().size() == 1) {
          out_start = out->lod()[0][out_offset];
        }
        for (int j = 0; j < repeat_num; j++) {
          for (int k = 0; k < x_seq_len; k++) {
            phi::memory_utils::Copy(
                gpu_place,
                out_data + (out_start + j * x_seq_len + k) * x_item_length,
                gpu_place,
                x_data + (x_start + k) * x_item_length,
                sizeof(T) * x_item_length,
                context.stream());
          }
        }
      } else {
        num_copies += repeat_num * x_seq_len;
      }
    }
    out_offset += repeat_num;
  }
  return num_copies;
}

template <typename T>
inline __global__ void sequence_expand_kernel(const T* x_data,
                                              const size_t* x_lod,
                                              const size_t* ref_lod,
                                              const size_t* offset,
                                              const size_t lod_size,
                                              /* default=1,
                                                 the instance length*/
                                              const int x_item_length,
                                              T* out_data) {
  int bid = blockIdx.x;
  if (bid >= lod_size - 1) return;

  int x_item_count = x_lod[bid + 1] - x_lod[bid];
  int repeats = ref_lod[bid + 1] - ref_lod[bid];
  int out_offset = static_cast<int>(offset[bid]);
  int x_offset = x_lod[bid];
  for (int tid_z = threadIdx.z; tid_z < repeats; tid_z += blockDim.z) {
    for (int tid_y = threadIdx.y; tid_y < x_item_count; tid_y += blockDim.y) {
      for (int tid_x = threadIdx.x; tid_x < x_item_length;
           tid_x += blockDim.x) {
        out_data[(out_offset + tid_z * x_item_count + tid_y) * x_item_length +
                 tid_x] = x_data[(x_offset + tid_y) * x_item_length + tid_x];
      }
    }
  }
}

template <typename T>
struct SequenceExpandFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& x,
                  const phi::Vector<size_t>& x_lod,   /*expand source lod*/
                  const phi::Vector<size_t>& ref_lod, /*expand referenced lod*/
                  DenseTensor* out) {
    int num_copies =
        ExpandByMemoryCopy<T>(context, x, out, x_lod, ref_lod, false);
    // Sometimes direct copies will be faster, this maybe need deeply analysis.
    if (num_copies < 5) {
      ExpandByMemoryCopy<T>(context, x, out, x_lod, ref_lod, true);
    } else {
      int x_item_length = x.numel() / x.dims()[0];
      size_t x_lod_size = x_lod.size();
      phi::Vector<size_t> out_offset(x_lod_size * 2 + ref_lod.size());
      GetOutputOffset(x_lod, ref_lod, &out_offset);

      for (size_t i = 0; i < x_lod_size; ++i) {
        out_offset[x_lod_size + i] = x_lod[i];
      }
      for (size_t i = 0; i < ref_lod.size(); ++i) {
        out_offset[2 * x_lod_size + i] = ref_lod[i];
      }

      phi::MixVector<size_t> mixv_out_offset(&out_offset);
      const size_t* out_offset_data =
          mixv_out_offset.CUDAData(context.GetPlace());
      const size_t* x_lod_data = out_offset_data + x_lod_size;
      const size_t* ref_lod_data = out_offset_data + 2 * x_lod_size;

      int thread_x =
          std::min(32, std::max(static_cast<int>(ref_lod.size()), 16));
      int thread_y = 16;
      int thread_z = 1024 / thread_x / thread_y;
      int block_x = static_cast<int>(ref_lod.size());
      dim3 block_size(thread_x, thread_y, thread_z);
      dim3 grid_size(block_x, 1);

      sequence_expand_kernel<<<grid_size, block_size, 0, context.stream()>>>(
          x.data<T>(),
          x_lod_data,
          ref_lod_data,
          out_offset_data,
          x_lod_size,
          x_item_length,
          context.template Alloc<T>(out));
    }
  }
};

}  // namespace phi
PD_REGISTER_KERNEL(sequence_expand,
                   GPU,
                   ALL_LAYOUT,
                   phi::SequenceExpandKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
