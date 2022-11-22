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

#include "paddle/phi/kernels/nms_kernel.h"

#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

static const int64_t threadsPerBlock = sizeof(int64_t) * 8;

namespace phi {

template <typename T>
static __global__ void NMS(const T* boxes_data,
                           float threshold,
                           int64_t num_boxes,
                           uint64_t* masks) {
  auto raw_start = blockIdx.y;
  auto col_start = blockIdx.x;
  if (raw_start > col_start) return;

  const int raw_last_storage =
      min(num_boxes - raw_start * threadsPerBlock, threadsPerBlock);
  const int col_last_storage =
      min(num_boxes - col_start * threadsPerBlock, threadsPerBlock);

  if (threadIdx.x < raw_last_storage) {
    uint64_t mask = 0;
    auto current_box_idx = raw_start * threadsPerBlock + threadIdx.x;
    const T* current_box = boxes_data + current_box_idx * 4;
    for (int i = 0; i < col_last_storage; ++i) {
      const T* target_box = boxes_data + (col_start * threadsPerBlock + i) * 4;
      if (CalculateIoU<T>(current_box, target_box, threshold)) {
        mask |= 1ULL << i;
      }
    }
    const int blocks_per_line = CeilDivide(num_boxes, threadsPerBlock);
    masks[current_box_idx * blocks_per_line + col_start] = mask;
  }
}

template <typename T, typename Context>
void NMSKernel(const Context& dev_ctx,
               const DenseTensor& boxes,
               float threshold,
               DenseTensor* output) {
  const int64_t num_boxes = boxes.dims()[0];
  const auto blocks_per_line = CeilDivide(num_boxes, threadsPerBlock);
  dim3 block(threadsPerBlock);
  dim3 grid(blocks_per_line, blocks_per_line);
  auto mask_data = paddle::memory::Alloc(
      dev_ctx.GetPlace(),
      num_boxes * blocks_per_line * sizeof(uint64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  uint64_t* mask_dev = reinterpret_cast<uint64_t*>(mask_data->ptr());
  NMS<T><<<grid, block, 0, dev_ctx.stream()>>>(
      boxes.data<T>(), threshold, num_boxes, mask_dev);
  std::vector<uint64_t> mask_host(num_boxes * blocks_per_line);
  paddle::memory::Copy(phi::CPUPlace(),
                       mask_host.data(),
                       dev_ctx.GetPlace(),
                       mask_dev,
                       num_boxes * blocks_per_line * sizeof(uint64_t),
                       dev_ctx.stream());
  std::vector<int64_t> remv(blocks_per_line);
  std::vector<int64_t> keep_boxes_idxs(num_boxes);
  int64_t* output_host = keep_boxes_idxs.data();
  int64_t last_box_num = 0;
  for (int64_t i = 0; i < num_boxes; ++i) {
    auto remv_element_id = i / threadsPerBlock;
    auto remv_bit_id = i % threadsPerBlock;
    if (!(remv[remv_element_id] & 1ULL << remv_bit_id)) {
      output_host[last_box_num++] = i;
      uint64_t* current_mask = mask_host.data() + i * blocks_per_line;
      for (auto j = remv_element_id; j < blocks_per_line; ++j) {
        remv[j] |= current_mask[j];
      }
    }
  }
  output->Resize(phi::make_ddim({last_box_num}));
  auto* output_data = dev_ctx.template Alloc<int64_t>(output);
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       output_data,
                       phi::CPUPlace(),
                       output_host,
                       sizeof(int64_t) * last_box_num,
                       dev_ctx.stream());
}
}  // namespace phi
PD_REGISTER_KERNEL(nms, GPU, ALL_LAYOUT, phi::NMSKernel, float, double) {}
