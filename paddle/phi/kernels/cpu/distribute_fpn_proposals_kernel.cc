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

#include "paddle/phi/kernels/distribute_fpn_proposals_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribute_fpn_proposals_functor.h"

namespace phi {

template <typename T, typename Context>
void DistributeFpnProposalsKernel(
    const Context& dev_ctx,
    const DenseTensor& fpn_rois,
    const paddle::optional<DenseTensor>& rois_num,
    int min_level,
    int max_level,
    int refer_level,
    int refer_scale,
    bool pixel_offset,
    std::vector<DenseTensor*> multi_fpn_rois,
    std::vector<DenseTensor*> multi_level_rois_num,
    DenseTensor* restore_index) {
  const int num_level = max_level - min_level + 1;

  // check that the fpn_rois is not empty
  if (!rois_num.get_ptr()) {
    PADDLE_ENFORCE_EQ(
        fpn_rois.lod().size(),
        1UL,
        errors::InvalidArgument("DistributeFpnProposalsOp needs LoD "
                                "with one level. But received level is %d",
                                fpn_rois.lod().size()));
  }

  std::vector<size_t> fpn_rois_lod;
  int fpn_rois_num;
  if (rois_num.get_ptr()) {
    fpn_rois_lod = funcs::GetLodFromRoisNum(dev_ctx, rois_num.get_ptr());
  } else {
    fpn_rois_lod = fpn_rois.lod().back();
  }
  fpn_rois_num = fpn_rois_lod[fpn_rois_lod.size() - 1];
  std::vector<int> target_level;

  // record the number of rois in each level
  std::vector<int> num_rois_level(num_level, 0);
  std::vector<int> num_rois_level_integral(num_level + 1, 0);
  for (size_t i = 0; i < fpn_rois_lod.size() - 1; ++i) {
    auto fpn_rois_slice = fpn_rois.Slice(fpn_rois_lod[i], fpn_rois_lod[i + 1]);
    const T* rois_data = fpn_rois_slice.data<T>();
    for (int j = 0; j < fpn_rois_slice.dims()[0]; ++j) {
      // get the target level of current rois
      T roi_scale = std::sqrt(funcs::BBoxArea(rois_data, pixel_offset));
      int tgt_lvl = std::floor(std::log2(roi_scale / refer_scale + (T)1e-6) +
                               refer_level);
      tgt_lvl = std::min(max_level, std::max(tgt_lvl, min_level));
      target_level.push_back(tgt_lvl);
      num_rois_level[tgt_lvl - min_level]++;
      rois_data += funcs::kBoxDim;
    }
  }
  // define the output rois
  // pointer which point to each level fpn rois
  std::vector<T*> multi_fpn_rois_data(num_level);
  // lod0 which will record the offset information of each level rois
  std::vector<std::vector<size_t>> multi_fpn_rois_lod0;
  for (int i = 0; i < num_level; ++i) {
    // allocate memory for each level rois
    multi_fpn_rois[i]->Resize({num_rois_level[i], funcs::kBoxDim});
    multi_fpn_rois_data[i] = dev_ctx.template Alloc<T>(multi_fpn_rois[i]);
    std::vector<size_t> lod0(1, 0);
    multi_fpn_rois_lod0.push_back(lod0);
    // statistic start point for each level rois
    num_rois_level_integral[i + 1] =
        num_rois_level_integral[i] + num_rois_level[i];
  }
  restore_index->Resize({fpn_rois_num, 1});
  int* restore_index_data = dev_ctx.template Alloc<int>(restore_index);
  std::vector<int> restore_index_inter(fpn_rois_num, -1);
  // distribute the rois into different fpn level by target level
  for (size_t i = 0; i < fpn_rois_lod.size() - 1; ++i) {
    auto fpn_rois_slice = fpn_rois.Slice(fpn_rois_lod[i], fpn_rois_lod[i + 1]);
    const T* rois_data = fpn_rois_slice.data<T>();
    size_t cur_offset = fpn_rois_lod[i];

    for (int j = 0; j < num_level; j++) {
      multi_fpn_rois_lod0[j].push_back(multi_fpn_rois_lod0[j][i]);
    }
    for (int j = 0; j < fpn_rois_slice.dims()[0]; ++j) {
      int lvl = target_level[cur_offset + j];
      memcpy(multi_fpn_rois_data[lvl - min_level],
             rois_data,
             funcs::kBoxDim * sizeof(T));
      multi_fpn_rois_data[lvl - min_level] += funcs::kBoxDim;
      int index_in_shuffle = num_rois_level_integral[lvl - min_level] +
                             multi_fpn_rois_lod0[lvl - min_level][i + 1];
      restore_index_inter[index_in_shuffle] = cur_offset + j;
      multi_fpn_rois_lod0[lvl - min_level][i + 1]++;
      rois_data += funcs::kBoxDim;
    }
  }
  for (int i = 0; i < fpn_rois_num; ++i) {
    restore_index_data[restore_index_inter[i]] = i;
  }

  if (multi_level_rois_num.size() > 0) {
    int batch_size = fpn_rois_lod.size() - 1;
    for (int i = 0; i < num_level; ++i) {
      multi_level_rois_num[i]->Resize({batch_size});
      int* rois_num_data = dev_ctx.template Alloc<int>(multi_level_rois_num[i]);
      for (int j = 0; j < batch_size; ++j) {
        rois_num_data[j] = static_cast<int>(multi_fpn_rois_lod0[i][j + 1] -
                                            multi_fpn_rois_lod0[i][j]);
      }
    }
  }
  // merge lod information into LoDTensor
  for (int i = 0; i < num_level; ++i) {
    LoD lod;
    lod.emplace_back(multi_fpn_rois_lod0[i]);
    multi_fpn_rois[i]->set_lod(lod);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(distribute_fpn_proposals,
                   CPU,
                   ALL_LAYOUT,
                   phi::DistributeFpnProposalsKernel,
                   float,
                   double) {}
