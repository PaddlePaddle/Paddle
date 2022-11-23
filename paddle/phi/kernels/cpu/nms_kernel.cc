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
#include "paddle/phi/backends/cpu/cpu_context.h"

#include "paddle/phi/core/kernel_registry.h"
<<<<<<< HEAD
=======
#include "paddle/phi/core/tensor_utils.h"
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
#include "paddle/phi/kernels/funcs/diagonal.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
<<<<<<< HEAD
static void NMS(const T* boxes_data,
                int64_t* output_data,
                float threshold,
                int64_t num_boxes) {
=======
static int64_t NMS(const T* boxes_data,
                   int64_t* output_data,
                   float threshold,
                   int64_t num_boxes) {
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  auto num_masks = CeilDivide(num_boxes, 64);
  std::vector<uint64_t> masks(num_masks, 0);

  for (int64_t i = 0; i < num_boxes; ++i) {
    if (masks[i / 64] & 1ULL << (i % 64)) continue;
    T box_1[4];
    for (int k = 0; k < 4; ++k) {
      box_1[k] = boxes_data[i * 4 + k];
    }
    for (int64_t j = i + 1; j < num_boxes; ++j) {
      if (masks[j / 64] & 1ULL << (j % 64)) continue;
      T box_2[4];
      for (int k = 0; k < 4; ++k) {
        box_2[k] = boxes_data[j * 4 + k];
      }
      bool is_overlap = CalculateIoU<T>(box_1, box_2, threshold);
      if (is_overlap) {
        masks[j / 64] |= 1ULL << (j % 64);
      }
    }
  }

  int64_t output_data_idx = 0;
  for (int64_t i = 0; i < num_boxes; ++i) {
    if (masks[i / 64] & 1ULL << (i % 64)) continue;
    output_data[output_data_idx++] = i;
  }

<<<<<<< HEAD
  for (; output_data_idx < num_boxes; ++output_data_idx) {
    output_data[output_data_idx] = 0;
  }
=======
  int64_t num_keep_boxes = output_data_idx;

  for (; output_data_idx < num_boxes; ++output_data_idx) {
    output_data[output_data_idx] = 0;
  }

  return num_keep_boxes;
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
}

template <typename T, typename Context>
void NMSKernel(const Context& dev_ctx,
               const DenseTensor& boxes,
               float threshold,
               DenseTensor* output) {
<<<<<<< HEAD
  auto output_data = dev_ctx.template Alloc<int64_t>(output);
  NMS<T>(boxes.data<T>(), output_data, threshold, boxes.dims()[0]);
=======
  int64_t num_boxes = boxes.dims()[0];
  DenseTensor output_tmp;
  output_tmp.Resize(phi::make_ddim({num_boxes}));
  auto output_tmp_data = dev_ctx.template Alloc<int64_t>(&output_tmp);

  int64_t num_keep_boxes =
      NMS<T>(boxes.data<T>(), output_tmp_data, threshold, num_boxes);
  auto slice_out = output_tmp.Slice(0, num_keep_boxes);
  phi::Copy(dev_ctx, slice_out, dev_ctx.GetPlace(), false, output);
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
}

}  // namespace phi

PD_REGISTER_KERNEL(nms, CPU, ALL_LAYOUT, phi::NMSKernel, float, double) {}
