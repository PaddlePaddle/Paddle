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

#include "paddle/phi/kernels/distribute_fpn_proposals_kernel.h"
#include "paddle/phi/kernels/funcs/distribute_fpn_proposals_functor.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
static void Sort(const XPUContext& dev_ctx,
                 const DenseTensor& value,
                 DenseTensor* index_out) {
  auto* value_data = value.data<T>();
  auto place = dev_ctx.GetPlace();
  auto cpu_place = phi::CPUPlace();

  DenseTensor scores_slice_cpu;
  scores_slice_cpu.Resize({value.numel()});
  T* scores_slice_cpu_data = dev_ctx.template HostAlloc<T>(&scores_slice_cpu);

  memory_utils::Copy(cpu_place,
                     scores_slice_cpu_data,
                     place,
                     value_data,
                     sizeof(T) * value.numel());
  // Sort index
  DenseTensor index_t;
  index_t.Resize({value.numel()});
  int* index = dev_ctx.template HostAlloc<int>(&index_t);
  for (int i = 0; i < value.numel(); ++i) {
    index[i] = i;
  }

  auto compare = [scores_slice_cpu_data](const int64_t& i, const int64_t& j) {
    return scores_slice_cpu_data[i] < scores_slice_cpu_data[j];
  };

  std::stable_sort(index, index + value.numel(), compare);
  index_out->Resize({index_t.numel()});
  int* idx_out = dev_ctx.template Alloc<int>(index_out);
  memory_utils::Copy(
      place, idx_out, cpu_place, index, sizeof(T) * index_t.numel());
}

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
        errors::InvalidArgument("DistributeFpnProposalsOp needs LoD"
                                "with one level"));
  }
  using XPUType = typename XPUTypeTrait<T>::Type;

  std::vector<size_t> fpn_rois_lod;
  if (rois_num.get_ptr()) {
    fpn_rois_lod = funcs::GetLodFromRoisNum(dev_ctx, rois_num.get_ptr());
  } else {
    fpn_rois_lod = fpn_rois.lod().back();
  }

  int lod_size = fpn_rois_lod.size() - 1;
  // the total num of roi
  int roi_num = fpn_rois_lod[lod_size];

  DenseTensor sub_lod_list;
  sub_lod_list.Resize({num_level, lod_size});
  int* sub_lod_list_data = dev_ctx.template Alloc<int>(&sub_lod_list);
  phi::funcs::SetConstant<phi::XPUContext, int> set_zero;
  set_zero(dev_ctx, &sub_lod_list, static_cast<int>(0));

  DenseTensor target_lvls;
  target_lvls.Resize({roi_num});
  int* target_lvls_data = dev_ctx.template Alloc<int>(&target_lvls);

  std::vector<int> rois_lod_vec(fpn_rois_lod.size(), 0);
  for (size_t i = 0; i < fpn_rois_lod.size(); ++i) {
    rois_lod_vec[i] = static_cast<int>(fpn_rois_lod[i]);
  }
  xpu::VectorParam<int> rois_lod = {
      rois_lod_vec.data(), static_cast<int>(rois_lod_vec.size()), nullptr};

  int r = xpu::distribute_fpn_proposals_helper<XPUType, int>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(fpn_rois.data<T>()),
      rois_lod,
      sub_lod_list_data,
      target_lvls_data,
      static_cast<int64_t>(min_level),
      static_cast<int64_t>(max_level),
      static_cast<int64_t>(refer_level),
      static_cast<int64_t>(refer_scale),
      pixel_offset);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "distribute_fpn_proposals_helper");

  DenseTensor index_out_t;
  Sort<int>(dev_ctx, target_lvls, &index_out_t);
  Sort<int>(dev_ctx, index_out_t, restore_index);
  restore_index->Resize({roi_num, 1});

  int start = 0;
  std::vector<int> sub_lod_list_cpu(lod_size * num_level);
  phi::TensorToVector<int>(sub_lod_list, dev_ctx, &sub_lod_list_cpu);

  for (int i = 0; i < num_level; ++i) {
    DenseTensor sub_lod = sub_lod_list.Slice(i, i + 1);
    // transfer length-based lod to offset-based lod
    std::vector<size_t> offset(1, 0);
    for (int j = 0; j < lod_size; ++j) {
      offset.emplace_back(offset.back() + sub_lod_list_cpu[i * lod_size + j]);
    }
    int sub_rois_num = offset.back();
    int end = start + sub_rois_num;
    if (end > start) {
      DenseTensor sub_idx = index_out_t.Slice(start, end);
      start = end;
      multi_fpn_rois[i]->Resize({sub_rois_num, funcs::kBoxDim});
      dev_ctx.template Alloc<T>(multi_fpn_rois[i]);

      std::vector<int64_t> fpn_rois_shape(fpn_rois.dims().size());
      for (int i = 0; i < fpn_rois.dims().size(); ++i) {
        fpn_rois_shape[i] = fpn_rois.dims()[i];
      }
      int r1 = xpu::paddle_gather<XPUType, int>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(fpn_rois.data<T>()),
          sub_idx.data<int>(),
          reinterpret_cast<XPUType*>(multi_fpn_rois[i]->data<T>()),
          fpn_rois_shape,
          sub_idx.numel(),
          0);
      PADDLE_ENFORCE_XDNN_SUCCESS(r1, "paddle_gather");
    } else {
      multi_fpn_rois[i]->Resize({sub_rois_num, funcs::kBoxDim});
      dev_ctx.template Alloc<T>(multi_fpn_rois[i]);
    }
    if (multi_level_rois_num.size() > 0) {
      DenseTensor* rois_num_t = multi_level_rois_num[i];
      Copy(dev_ctx, sub_lod, dev_ctx.GetPlace(), true, rois_num_t);
      rois_num_t->Resize({lod_size});
    }
    LoD lod;
    lod.emplace_back(offset);
    multi_fpn_rois[i]->set_lod(lod);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(distribute_fpn_proposals,
                   XPU,
                   ALL_LAYOUT,
                   phi::DistributeFpnProposalsKernel,
                   float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
