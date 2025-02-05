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

#include "paddle/phi/kernels/roi_align_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RoiAlignKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& boxes,
                    const paddle::optional<DenseTensor>& boxes_num,
                    int pooled_height,
                    int pooled_width,
                    float spatial_scale,
                    int sampling_ratio,
                    bool aligned,
                    DenseTensor* out) {
  const auto& in_dims = x.dims();
  int batch_size = in_dims[0];
  int channels = in_dims[1];
  int height = in_dims[2];
  int width = in_dims[3];

  int rois_num = boxes.dims()[0];

  if (rois_num == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  DenseTensor roi_batch_id_list;
  roi_batch_id_list.Resize({rois_num});
  auto cplace = phi::CPUPlace();
  int* roi_batch_id_data = dev_ctx.template HostAlloc<int>(&roi_batch_id_list);
  auto xplace = dev_ctx.GetPlace();
  int rois_batch_size = 0;
  int* cpu_lod = nullptr;
  if (boxes_num) {
    rois_batch_size = boxes_num->numel();
    PADDLE_ENFORCE_EQ(
        rois_batch_size,
        batch_size,
        errors::InvalidArgument(
            "The rois_batch_size and imgs "
            "batch_size must be the same. But received rois_batch_size = %d, "
            "batch_size = %d",
            rois_batch_size,
            batch_size));

    if (boxes_num->dtype() == phi::DataType::INT64) {
      std::vector<int64_t> rois_num_list(rois_batch_size);
      memory_utils::Copy(cplace,
                         rois_num_list.data(),
                         xplace,
                         boxes_num->data<int64_t>(),
                         sizeof(int64_t) * rois_batch_size);
      cpu_lod = new int[rois_batch_size + 1];
      cpu_lod[0] = 0;
      for (int64_t i = 0; i < rois_batch_size; i++) {
        cpu_lod[i + 1] = cpu_lod[i] + rois_num_list[i];
      }
    } else if (boxes_num->dtype() == phi::DataType::INT32) {
      std::vector<int> rois_num_list(rois_batch_size);
      memory_utils::Copy(cplace,
                         rois_num_list.data(),
                         xplace,
                         boxes_num->data<int>(),
                         sizeof(int) * rois_batch_size);
      cpu_lod = new int[rois_batch_size + 1];
      cpu_lod[0] = 0;
      for (int i = 0; i < rois_batch_size; i++) {
        cpu_lod[i + 1] = cpu_lod[i] + rois_num_list[i];
      }
    }
  } else {
    auto lod = boxes.lod();
    PADDLE_ENFORCE_EQ(lod.empty(),
                      false,
                      errors::InvalidArgument("Input(ROIs) in ROIAlignOp does "
                                              "not contain LoD information."));
    auto rois_lod = lod.back();
    rois_batch_size = rois_lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        rois_batch_size,
        batch_size,
        errors::InvalidArgument(
            "The batch size of rois and batch size "
            "of images must be the same. But received rois batch size = %d, "
            "and images batch size = %d",
            rois_batch_size,
            batch_size));
    int rois_num_with_lod = rois_lod[rois_batch_size];
    PADDLE_ENFORCE_EQ(
        rois_num,
        rois_num_with_lod,
        errors::InvalidArgument(
            "The actual number of rois and the number of rois "
            "provided from Input(RoIsLoD) in RoIAlign must be the same."
            " But received actual number of rois is %d, and the number "
            "of rois from RoIsLoD is %d",
            rois_num,
            rois_num_with_lod));
    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
    cpu_lod = new int[rois_batch_size + 1];
    for (int i = 0; i < rois_batch_size + 1; i++) {
      cpu_lod[i] = rois_lod[i];
    }
  }

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int* roi_id_data = RAII_GUARD.alloc_l3_or_gm<int>(rois_batch_size + 1);
  PADDLE_ENFORCE_NOT_NULL(
      roi_id_data, errors::ResourceExhausted("XPU has no enough memory"));
  memory_utils::Copy(xplace,
                     roi_id_data,
                     cplace,
                     cpu_lod,
                     (rois_batch_size + 1) * sizeof(int));
  delete[] cpu_lod;
  int r = xpu::roi_align<T, int>(dev_ctx.x_context(),
                                 x.data<T>(),
                                 dev_ctx.template Alloc<T>(out),
                                 boxes.data<T>(),
                                 roi_id_data,
                                 batch_size,
                                 channels,
                                 height,
                                 width,
                                 out->dims()[0],
                                 pooled_height,
                                 pooled_width,
                                 spatial_scale,
                                 sampling_ratio,
                                 true,
                                 aligned);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "roi_align");
}

}  // namespace phi

PD_REGISTER_KERNEL(roi_align, XPU, ALL_LAYOUT, phi::RoiAlignKernel, float) {}
