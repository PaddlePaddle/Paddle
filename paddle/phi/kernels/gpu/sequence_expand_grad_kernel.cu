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
inline __global__ void sequence_expand_grad_kernel(const T* dout_data,
                                                   const size_t* ref_lod,
                                                   const size_t* dx_lod,
                                                   const size_t* offset,
                                                   const size_t lod_size,
                                                   /* default=1,
                                                      the instance length*/
                                                   const int x_item_length,
                                                   T* dx_data) {
  int bid = blockIdx.x;
  if (bid >= lod_size - 1) return;
  int x_item_count = dx_lod[bid + 1] - dx_lod[bid];
  int repeats = ref_lod[bid + 1] - ref_lod[bid];
  int out_offset = static_cast<int>(offset[bid]);
  int x_offset = dx_lod[bid];

  for (int tid_z = threadIdx.z; tid_z < repeats; tid_z += blockDim.z) {
    for (int tid_y = threadIdx.y; tid_y < x_item_count; tid_y += blockDim.y) {
      for (int tid_x = threadIdx.x; tid_x < x_item_length;
           tid_x += blockDim.x) {
        phi::CudaAtomicAdd(
            &dx_data[(x_offset + tid_y) * x_item_length + tid_x],
            dout_data[(out_offset + tid_z * x_item_count + tid_y) *
                          x_item_length +
                      tid_x]);
      }
    }
  }
}

template <typename T>
struct SequenceExpandGradFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& dout,
                  const phi::Vector<size_t>& x_lod,   /*expand source lod*/
                  const phi::Vector<size_t>& ref_lod, /*expand based lod*/
                  DenseTensor* dx) {
    int x_item_length = common::product(dx->dims()) / dx->dims()[0];
    phi::Vector<size_t> out_offset(x_lod.size());
    GetOutputOffset(x_lod, ref_lod, &out_offset);

    int thread_x = std::min(32, std::max(static_cast<int>(ref_lod.size()), 16));
    int thread_y = 16;
    int thread_z = 1024 / thread_x / thread_y;
    int block_x = static_cast<int>(ref_lod.size());
    dim3 block_size(thread_x, thread_y, thread_z);
    dim3 grid_size(block_x, 1);
    phi::MixVector<size_t> mixv_ref_lod(&ref_lod);
    phi::MixVector<size_t> mixv_x_lod(&x_lod);
    phi::MixVector<size_t> mixv_out_offset(&out_offset);
    sequence_expand_grad_kernel<<<grid_size, block_size, 0, context.stream()>>>(
        dout.data<T>(),
        mixv_ref_lod.CUDAData(context.GetPlace()),
        mixv_x_lod.CUDAData(context.GetPlace()),
        mixv_out_offset.CUDAData(context.GetPlace()),
        ref_lod.size(),
        x_item_length,
        context.template Alloc<T>(dx));
  }
};

}  // namespace phi
PD_REGISTER_KERNEL(sequence_expand_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SequenceExpandGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
