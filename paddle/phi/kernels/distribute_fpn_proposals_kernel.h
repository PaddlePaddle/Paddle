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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

const int kBoxDim = 4;

template <typename Context>
inline std::vector<size_t> GetLodFromRoisNum(const Context& ctx,
                                             const DenseTensor* rois_num) {
  std::vector<size_t> rois_lod;
  auto* rois_num_data = rois_num->data<int>();
  DenseTensor cpu_tensor;
  if (rois_num->place().GetType() == phi::AllocationType::GPU) {
    phi::Copy(ctx, *rois_num, phi::CPUPlace(), true, &cpu_tensor);
    rois_num_data = cpu_tensor.data<int>();
  }
  rois_lod.push_back(static_cast<size_t>(0));
  for (int i = 0; i < rois_num->numel(); ++i) {
    rois_lod.push_back(rois_lod.back() + static_cast<size_t>(rois_num_data[i]));
  }
  return rois_lod;
}

template <typename T, typename Context>
void DistributeFpnProposalsKernel(
    const Context& ctx,
    const DenseTensor& fpnrois,
    const DenseTensor& roisnum,
    int min_level,
    int max_level,
    int refer_level,
    int refer_scale,
    bool pixel_offset,
    std::vector<DenseTensor*> multi_fpn_rois,
    DenseTensor* restore_index,
    std::vector<DenseTensor*> multi_level_roisnum);

}  // namespace phi
