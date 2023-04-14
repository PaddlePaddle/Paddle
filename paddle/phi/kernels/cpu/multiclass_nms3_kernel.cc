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

#include "paddle/phi/kernels/multiclass_nms3_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/multiclass_nms3_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MultiClassNMSKernel(const Context& ctx,
                         const DenseTensor& bboxes,
                         const DenseTensor& scores,
                         const paddle::optional<DenseTensor>& rois_num,
                         float score_threshold,
                         int nms_top_k,
                         int keep_top_k,
                         float nms_threshold,
                         bool normalized,
                         float nms_eta,
                         int background_label,
                         DenseTensor* out,
                         DenseTensor* index,
                         DenseTensor* nms_rois_num) {
  MultiClassNMSCPUKernel<T, Context>(ctx,
                                     bboxes,
                                     scores,
                                     rois_num,
                                     score_threshold,
                                     nms_top_k,
                                     keep_top_k,
                                     nms_threshold,
                                     normalized,
                                     nms_eta,
                                     background_label,
                                     out,
                                     index,
                                     nms_rois_num);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    multiclass_nms3, CPU, ALL_LAYOUT, phi::MultiClassNMSKernel, float, double) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
