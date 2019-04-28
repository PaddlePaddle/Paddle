/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

const int kBoxDim = 4;

template <typename T>
static inline T BBoxArea(const T* box, bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<T>(0.);
  } else {
    const T w = box[2] - box[0];
    const T h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

template <typename T>
class DistributeFpnProposalsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* fpn_rois = context.Input<paddle::framework::LoDTensor>("FpnRois");

    auto multi_fpn_rois =
        context.MultiOutput<paddle::framework::LoDTensor>("MultiFpnRois");

    auto* restore_index =
        context.Output<paddle::framework::Tensor>("RestoreIndex");

    const int min_level = context.Attr<int>("min_level");
    const int max_level = context.Attr<int>("max_level");
    const int refer_level = context.Attr<int>("refer_level");
    const int refer_scale = context.Attr<int>("refer_scale");
    const int num_level = max_level - min_level + 1;

    // check that the fpn_rois is not empty
    PADDLE_ENFORCE_EQ(fpn_rois->lod().size(), 1UL,
                      "DistributeFpnProposalsOp need 1 level of LoD");

    auto fpn_rois_lod = fpn_rois->lod().back();
    int fpn_rois_num = fpn_rois_lod[fpn_rois_lod.size() - 1];
    std::vector<int> target_level;
    // std::vector<int> target_level(fpn_rois_num, -1);
    // record the number of rois in each level
    std::vector<int> num_rois_level(num_level, 0);
    std::vector<int> num_rois_level_integral(num_level + 1, 0);
    for (size_t i = 0; i < fpn_rois_lod.size() - 1; ++i) {
      Tensor fpn_rois_slice =
          fpn_rois->Slice(fpn_rois_lod[i], fpn_rois_lod[i + 1]);
      const T* rois_data = fpn_rois_slice.data<T>();
      for (int j = 0; j < fpn_rois_slice.dims()[0]; ++j) {
        // get the target level of current rois
        T roi_scale = std::sqrt(BBoxArea(rois_data, false));
        int tgt_lvl =
            std::floor(std::log2(roi_scale / refer_scale) + refer_level);
        tgt_lvl = std::min(max_level, std::max(tgt_lvl, min_level));
        target_level.push_back(tgt_lvl);
        num_rois_level[tgt_lvl - min_level]++;
        rois_data += kBoxDim;
      }
    }
    // define the output rois
    // pointer which point to each level fpn rois
    std::vector<T*> multi_fpn_rois_data(num_level);
    // lod0 which will record the offset information of each level rois
    std::vector<std::vector<size_t>> multi_fpn_rois_lod0;
    for (int i = 0; i < num_level; ++i) {
      // allocate memory for each level rois
      multi_fpn_rois[i]->mutable_data<T>({num_rois_level[i], kBoxDim},
                                         context.GetPlace());
      multi_fpn_rois_data[i] = multi_fpn_rois[i]->data<T>();
      std::vector<size_t> lod0(1, 0);
      multi_fpn_rois_lod0.push_back(lod0);
      // statistic start point for each level rois
      num_rois_level_integral[i + 1] =
          num_rois_level_integral[i] + num_rois_level[i];
    }
    restore_index->mutable_data<int>({1, fpn_rois_num}, context.GetPlace());
    int* restore_index_data = restore_index->data<int>();
    std::vector<int> restore_index_inter(fpn_rois_num, -1);
    // distribute the rois into different fpn level by target level
    for (size_t i = 0; i < fpn_rois_lod.size() - 1; ++i) {
      Tensor fpn_rois_slice =
          fpn_rois->Slice(fpn_rois_lod[i], fpn_rois_lod[i + 1]);
      const T* rois_data = fpn_rois_slice.data<T>();
      size_t cur_offset = fpn_rois_lod[i];
      // std::vector<size_t > lod_offset[num_level];
      for (int j = 0; j < num_level; j++) {
        multi_fpn_rois_lod0[j].push_back(multi_fpn_rois_lod0[j][i]);
      }
      for (int j = 0; j < fpn_rois_slice.dims()[0]; ++j) {
        int lvl = target_level[cur_offset + j];
        memcpy(multi_fpn_rois_data[lvl - min_level], rois_data,
               kBoxDim * sizeof(T));
        multi_fpn_rois_data[lvl - min_level] += kBoxDim;
        int index_in_shuffle = num_rois_level_integral[lvl - min_level] +
                               multi_fpn_rois_lod0[lvl - min_level][i + 1];
        restore_index_inter[index_in_shuffle] = cur_offset + j;
        multi_fpn_rois_lod0[lvl - min_level][i + 1]++;
        rois_data += kBoxDim;
      }
    }
    for (int i = 0; i < fpn_rois_num; ++i) {
      restore_index_data[restore_index_inter[i]] = i;
    }
    // merge lod information into LoDTensor
    for (int i = 0; i < num_level; ++i) {
      framework::LoD lod;
      lod.emplace_back(multi_fpn_rois_lod0[i]);
      multi_fpn_rois[i]->set_lod(lod);
    }
  }
};
}  // namespace operators
}  // namespace paddle
