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

#include "paddle/phi/kernels/roi_pool_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void RoiPoolGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& boxes,
                       paddle::optional<const DenseTensor&> boxes_num,
                       const DenseTensor& arg_max,
                       const DenseTensor& out_grad,
                       int pooled_height,
                       int pooled_width,
                       float spatial_scale,
                       DenseTensor* dx) {
  if (dx) {
    int rois_num = boxes.dims()[0];
    DenseTensor box_batch_id_list = Empty<int>(dev_ctx, {rois_num});
    int* box_batch_id_data = box_batch_id_list.data<int>();

    int boxes_batch_size;
    if (boxes_num) {
      boxes_batch_size = boxes_num->numel();
      auto* boxes_num_data = boxes_num->data<int>();
      int start = 0;
      for (int n = 0; n < boxes_batch_size; ++n) {
        for (int i = start; i < start + boxes_num_data[n]; ++i) {
          box_batch_id_data[i] = n;
        }
        start += boxes_num_data[n];
      }
    } else {
      auto boxes_lod = boxes.lod().back();
      boxes_batch_size = boxes_lod.size() - 1;
      for (int n = 0; n < boxes_batch_size; ++n) {
        for (size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
          box_batch_id_data[i] = n;
        }
      }
    }

    const T* boxes_data = boxes.data<T>();
    const T* out_grad_data = out_grad.data<T>();
    const int64_t* arg_max_data = arg_max.data<int64_t>();
    T* dx_data = dev_ctx.template Alloc<T>(dx);

    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, dx, static_cast<T>(0));

    auto in_stride = phi::stride(x.dims());
    auto arg_max_stride = phi::stride(arg_max.dims());
    auto roi_stride = phi::stride(boxes.dims());
    auto out_stride = phi::stride(out_grad.dims());

    int channels = x.dims()[1];

    for (int n = 0; n < rois_num; ++n) {
      int roi_batch_idx = box_batch_id_data[n];
      T* batch_grad_data = dx_data + roi_batch_idx * in_stride[0];
      for (int c = 0; c < channels; ++c) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          for (int pw = 0; pw < pooled_width; ++pw) {
            int pool_index = ph * pooled_width + pw;
            if (arg_max_data[pool_index] >= 0) {
              auto index = arg_max_data[pool_index];
              batch_grad_data[index] += out_grad_data[pool_index];
            }
          }
        }
        batch_grad_data += in_stride[1];
        out_grad_data += out_stride[1];
        arg_max_data += arg_max_stride[1];
      }
      boxes_data += roi_stride[0];
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(roi_pool_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::RoiPoolGradKernel,
                   float,
                   double,
                   int) {
  kernel->InputAt(3).SetDataType(phi::DataType::INT64);
}
