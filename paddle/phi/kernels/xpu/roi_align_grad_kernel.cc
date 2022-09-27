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

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RoiAlignGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& boxes,
                        const paddle::optional<DenseTensor>& boxes_num,
                        const DenseTensor& out_grad,
                        int pooled_height,
                        int pooled_width,
                        float spatial_scale,
                        int sampling_ratio,
                        bool aligned,
                        DenseTensor* dx) {
  int rois_num = boxes.dims()[0];
  int channels = x.dims()[1];
  int height = x.dims()[2];
  int width = x.dims()[3];

  if (!dx) {
    return;
  }
  DenseTensor roi_batch_id_list;
  roi_batch_id_list.Resize({rois_num});
  auto cplace = phi::CPUPlace();
  auto xplace = dev_ctx.GetPlace();

  int rois_batch_size = 0;
  int* cpu_lod = nullptr;
  if (boxes_num) {
    rois_batch_size = boxes_num->numel();
    std::vector<int> rois_num_list(rois_batch_size);
    paddle::memory::Copy(cplace,
                         rois_num_list.data(),
                         xplace,
                         boxes_num->data<int>(),
                         sizeof(int) * rois_batch_size);
    cpu_lod = new int[rois_batch_size + 1];
    cpu_lod[0] = 0;
    for (int i = 0; i < rois_batch_size; i++) {
      cpu_lod[i + 1] = cpu_lod[i] + rois_num_list[i];
    }
  } else {
    auto rois_lod = boxes.lod().back();
    rois_batch_size = rois_lod.size() - 1;
    cpu_lod = new int[rois_batch_size + 1];
    for (int i = 0; i < rois_batch_size + 1; i++) {
      cpu_lod[i] = rois_lod[i];
    }
  }
  int* roi_id_data = nullptr;
  int r = xpu_malloc(reinterpret_cast<void**>(&roi_id_data),
                     (rois_batch_size + 1) * sizeof(int));
  PADDLE_ENFORCE_XPU_SUCCESS(r);
  paddle::memory::Copy(xplace,
                       roi_id_data,
                       cplace,
                       cpu_lod,
                       (rois_batch_size + 1) * sizeof(int));
  dev_ctx.template Alloc<T>(dx);

  int output_grad_size = out_grad.numel();

  delete[] cpu_lod;
  if (output_grad_size > 0) {
    r = xpu::roi_align_grad<T, int>(dev_ctx.x_context(),
                                    out_grad.data<T>(),
                                    dx->data<T>(),
                                    boxes.data<T>(),
                                    roi_id_data,
                                    x.dims()[0],
                                    channels,
                                    height,
                                    width,
                                    out_grad.dims()[0],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    sampling_ratio,
                                    true,
                                    aligned);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "roi_align_grad");
  }
  if (dev_ctx.x_context()->xpu_stream) {
    dev_ctx.Wait();
  }
  xpu_free(roi_id_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    roi_align_grad, XPU, ALL_LAYOUT, phi::RoiAlignGradKernel, float) {}
