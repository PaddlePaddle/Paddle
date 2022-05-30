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
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

const int kBoxDim = 4;

inline std::vector<size_t> GetLodFromRoisNum(
    const framework::Tensor* rois_num) {
  std::vector<size_t> rois_lod;
  auto* rois_num_data = rois_num->data<int>();
  framework::Tensor cpu_tensor;
  if (platform::is_gpu_place(rois_num->place())) {
    paddle::framework::TensorCopySync(*rois_num, platform::CPUPlace(),
                                      &cpu_tensor);
    rois_num_data = cpu_tensor.data<int>();
  }
  rois_lod.push_back(static_cast<size_t>(0));
  for (int i = 0; i < rois_num->numel(); ++i) {
    rois_lod.push_back(rois_lod.back() + static_cast<size_t>(rois_num_data[i]));
  }
  return rois_lod;
}

template <typename T>
static inline T BBoxArea(const T* box, bool pixel_offset) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<T>(0.);
  } else {
    const T w = box[2] - box[0];
    const T h = box[3] - box[1];
    if (pixel_offset) {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    } else {
      return w * h;
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
    const bool pixel_offset = context.Attr<bool>("pixel_offset");
    const int num_level = max_level - min_level + 1;

    // check that the fpn_rois is not empty
    if (!context.HasInput("RoisNum")) {
      PADDLE_ENFORCE_EQ(fpn_rois->lod().size(), 1UL,
                        platform::errors::InvalidArgument(
                            "DistributeFpnProposalsOp needs LoD "
                            "with one level. But received level is %d",
                            fpn_rois->lod().size()));
    }

    std::vector<size_t> fpn_rois_lod;
    int fpn_rois_num;
    if (context.HasInput("RoisNum")) {
      auto* rois_num = context.Input<framework::Tensor>("RoisNum");
      fpn_rois_lod = GetLodFromRoisNum(rois_num);
    } else {
      fpn_rois_lod = fpn_rois->lod().back();
    }
    fpn_rois_num = fpn_rois_lod[fpn_rois_lod.size() - 1];
    std::vector<int> target_level;
    // std::vector<int> target_level(fpn_rois_num, -1);
    // record the number of rois in each level
    std::vector<int> num_rois_level(num_level, 0);
    std::vector<int> num_rois_level_integral(num_level + 1, 0);
    for (size_t i = 0; i < fpn_rois_lod.size() - 1; ++i) {
      auto fpn_rois_slice =
          fpn_rois->Slice(fpn_rois_lod[i], fpn_rois_lod[i + 1]);
      const T* rois_data = fpn_rois_slice.data<T>();
      for (int j = 0; j < fpn_rois_slice.dims()[0]; ++j) {
        // get the target level of current rois
        T roi_scale = std::sqrt(BBoxArea(rois_data, pixel_offset));
        int tgt_lvl = std::floor(std::log2(roi_scale / refer_scale + (T)1e-6) +
                                 refer_level);
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
    restore_index->mutable_data<int>({fpn_rois_num, 1}, context.GetPlace());
    int* restore_index_data = restore_index->data<int>();
    std::vector<int> restore_index_inter(fpn_rois_num, -1);
    // distribute the rois into different fpn level by target level
    for (size_t i = 0; i < fpn_rois_lod.size() - 1; ++i) {
      auto fpn_rois_slice =
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
    auto multi_rois_num =
        context.MultiOutput<framework::Tensor>("MultiLevelRoIsNum");
    if (multi_rois_num.size() > 0) {
      int batch_size = fpn_rois_lod.size() - 1;
      for (int i = 0; i < num_level; ++i) {
        int* rois_num_data = multi_rois_num[i]->mutable_data<int>(
            {batch_size}, context.GetPlace());
        for (int j = 0; j < batch_size; ++j) {
          rois_num_data[j] = static_cast<int>(multi_fpn_rois_lod0[i][j + 1] -
                                              multi_fpn_rois_lod0[i][j]);
        }
      }
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
