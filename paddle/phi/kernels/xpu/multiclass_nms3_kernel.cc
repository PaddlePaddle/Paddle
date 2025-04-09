// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MultiClassNMSKernel(const Context& ctx,
                         const DenseTensor& bboxes,
                         const DenseTensor& scores,
                         const paddle::optional<DenseTensor>& rois_num,
                         float score_threshold,
                         int nums_top_k,
                         int keep_top_k,
                         float nms_threshold,
                         bool normalized,
                         float nms_eta,
                         int background_label,
                         DenseTensor* out,
                         DenseTensor* index,
                         DenseTensor* nms_rois_num) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const XPUType* bboxes_data =
      reinterpret_cast<const XPUType*>(bboxes.data<T>());
  const XPUType* scores_data =
      reinterpret_cast<const XPUType*>(scores.data<T>());

  bool return_index = index != nullptr;
  bool has_rois_num = rois_num.get_ptr() != nullptr;
  bool return_rois_num = nms_rois_num != nullptr;
  auto score_dims = common::vectorize<int>(scores.dims());
  auto score_size = score_dims.size();
  bool is_lod = score_size == 2 ? true : false;

  int n = 0;
  int b = 0;
  int class_num = scores.dims()[1];
  int out_dim = bboxes.dims()[2] + 2;
  int boxes_count = 0;
  std::vector<int> rois_num_vec;
  rois_num_vec.clear();
  if (is_lod) {
    if (has_rois_num) {
      phi::DenseTensor rois_num_host;
      rois_num_host.Resize(rois_num.get_ptr()->dims());
      if (rois_num.get_ptr()->dtype() == phi::DataType::INT64) {
        ctx.template HostAlloc<int64_t>(&rois_num_host);
        phi::Copy(ctx,
                  *rois_num.get_ptr(),
                  rois_num_host.place(),
                  false,
                  &rois_num_host);
        n = rois_num.get_ptr()->numel();
        for (int64_t i = 0; i < n; i++) {
          rois_num_vec.push_back(rois_num_host.data<int64_t>()[i]);
          boxes_count += rois_num_host.data<int64_t>()[i];
        }
      } else if (rois_num.get_ptr()->dtype() == phi::DataType::INT32) {
        ctx.template HostAlloc<int>(&rois_num_host);
        phi::Copy(ctx,
                  *rois_num.get_ptr(),
                  rois_num_host.place(),
                  false,
                  &rois_num_host);
        n = rois_num.get_ptr()->numel();
        for (int i = 0; i < n; i++) {
          rois_num_vec.push_back(rois_num_host.data<int>()[i]);
          boxes_count += rois_num_host.data<int>()[i];
        }
      }
    } else {
      auto lod = bboxes.lod().back();
      boxes_count = lod[lod.size() - 1];
      n = lod.size() - 1;
      for (int i = 0; i < n; i++) {
        rois_num_vec.push_back(lod[i + 1] - lod[i]);
      }
    }
    PADDLE_ENFORCE_EQ(boxes_count == bboxes.dims()[0],
                      true,
                      common::errors::InvalidArgument(
                          "boxes_count should equal boxes->dims()[0].",
                          "But received: (%d) and (%d)",
                          boxes_count,
                          bboxes.dims()[0]));
    PADDLE_ENFORCE_EQ(boxes_count == score_dims[0],
                      true,
                      common::errors::InvalidArgument(
                          "boxes_count should equal score_dims[0].",
                          "But received: (%d) and (%d)",
                          boxes_count,
                          score_dims[0]));
  } else {
    n = bboxes.dims()[0];
    b = bboxes.dims()[1];
    boxes_count = n * b;
  }
  std::vector<T> outs_vec_;
  std::vector<int> out_index_vec_;

  outs_vec_.resize(boxes_count * out_dim);
  out_index_vec_.resize(boxes_count);

  std::vector<size_t> batch_starts;
  int r = 0;
  r = xpu::multiclass_nms<T, int>(ctx.x_context(),
                                  bboxes_data,
                                  scores_data,
                                  rois_num_vec,
                                  outs_vec_,
                                  out_index_vec_,
                                  batch_starts,
                                  n,
                                  b,
                                  class_num,
                                  out_dim,
                                  nums_top_k,
                                  score_threshold,
                                  keep_top_k,
                                  nms_threshold,
                                  background_label,
                                  normalized,
                                  nms_eta,
                                  return_index,
                                  is_lod);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "multiclass_nms");
  uint64_t num_kept = batch_starts.back();

  if (num_kept == 0) {
    if (return_index) {
      // out_dim may be zero when there is no object in picture, so add some
      // zeros to it
      // caution: results may differ between cpu and xpu due to this operation
      out->Resize({1, out_dim});
      ctx.template Alloc<T>(out);
      T* out_ptr = out->template data<T>();
      std::vector<T> temp_value(out_dim, 0.0f);
      memory_utils::Copy(ctx.GetPlace(),
                         out_ptr,
                         phi::CPUPlace(),
                         temp_value.data(),
                         1 * out_dim * sizeof(T));

      index->Resize({1, 1});
      ctx.template Alloc<int>(index);
      int* out_index_ptr = index->template data<int>();
      std::vector<int> temp_idx(1, 0);
      memory_utils::Copy(ctx.GetPlace(),
                         out_index_ptr,
                         phi::CPUPlace(),
                         temp_idx.data(),
                         1 * sizeof(int));
    } else {
      out->Resize({1, 1});
      T* od = ctx.template Alloc<T>(out);
      od[0] = -1;
      batch_starts = {0, 1};
    }
  } else {
    out->Resize({static_cast<int64_t>(num_kept), out_dim});
    ctx.template Alloc<T>(out);
    T* out_ptr = out->template data<T>();
    memory_utils::Copy(ctx.GetPlace(),
                       out_ptr,
                       phi::CPUPlace(),
                       outs_vec_.data(),
                       num_kept * out_dim * sizeof(T));
    if (return_index) {
      index->Resize({static_cast<int64_t>(num_kept), 1});
      ctx.template Alloc<int>(index);
      int* out_index_ptr = index->template data<int>();
      memory_utils::Copy(ctx.GetPlace(),
                         out_index_ptr,
                         phi::CPUPlace(),
                         out_index_vec_.data(),
                         num_kept * sizeof(int));
    }
  }

  if (return_rois_num) {
    nms_rois_num->Resize({n});
    ctx.template Alloc<int>(nms_rois_num);

    DenseTensor nms_rois_num_cpu;
    nms_rois_num_cpu.Resize({nms_rois_num->numel()});
    ctx.template HostAlloc<int>(&nms_rois_num_cpu);
    int* nms_rois_num_cpu_data = nms_rois_num_cpu.data<int>();

    for (int i = 1; i <= n; i++) {
      nms_rois_num_cpu_data[i - 1] = batch_starts[i] - batch_starts[i - 1];
    }
    phi::Copy(ctx, nms_rois_num_cpu, nms_rois_num->place(), true, nms_rois_num);
  }
  LegacyLoD lod;
  if (num_kept == 0) {
    batch_starts[batch_starts.size() - 1] = 1;
  }
  lod.emplace_back(batch_starts);
  if (return_index) {
    index->set_lod(lod);
  }
  out->set_lod(lod);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    multiclass_nms3, XPU, ALL_LAYOUT, phi::MultiClassNMSKernel, float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
