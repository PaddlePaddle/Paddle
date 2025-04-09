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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/kernels/distribute_fpn_proposals_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/detection/bbox_util.h"
#include "paddle/phi/kernels/funcs/distribute_fpn_proposals_functor.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/phi/common/memory_utils.h"

namespace phi {

static constexpr int kNumCUDAThreads = 64;
static constexpr int kNumMaximumNumBlocks = 4096;

int const BBoxSize = 4;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <class T>
__global__ void GPUDistFpnProposalsHelper(const int nthreads,
                                          const T* rois,
                                          const int lod_size,
                                          const int refer_level,
                                          const int refer_scale,
                                          const int max_level,
                                          const int min_level,
                                          int* roi_batch_id_data,
                                          int* sub_lod_list,
                                          int* target_lvls,
                                          bool pixel_offset = true) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const T* offset_roi = rois + i * BBoxSize;
    int roi_batch_ind = roi_batch_id_data[i];
    // get the target level of current rois
    T roi_area;
    if (offset_roi[2] < offset_roi[0] || offset_roi[3] < offset_roi[1]) {
      roi_area = static_cast<T>(0.);
    } else {
      const T w = offset_roi[2] - offset_roi[0];
      const T h = offset_roi[3] - offset_roi[1];
      if (pixel_offset) {
        roi_area = (w + 1) * (h + 1);
      } else {
        roi_area = w * h;
      }
    }
    T roi_scale = sqrt(roi_area);
    int tgt_lvl = floor(
        log2(roi_scale / static_cast<T>(refer_scale) + (T)1e-8) + refer_level);
    tgt_lvl = min(max_level, max(tgt_lvl, min_level));
    target_lvls[i] = tgt_lvl;
    // compute number of rois in the same batch and same target level
    phi::CudaAtomicAdd(
        sub_lod_list + (tgt_lvl - min_level) * lod_size + roi_batch_ind, 1);
  }
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
  int num_level = max_level - min_level + 1;

  // check that the fpn_rois is not empty
  if (!rois_num.get_ptr()) {
    PADDLE_ENFORCE_EQ(
        fpn_rois.lod().size(),
        1UL,
        errors::InvalidArgument("DistributeFpnProposalsOp needs LoD"
                                "with one level"));
  }

  std::vector<size_t> fpn_rois_lod;
  if (rois_num.get_ptr()) {
    fpn_rois_lod = funcs::GetLodFromRoisNum(dev_ctx, rois_num.get_ptr());
  } else {
    fpn_rois_lod = fpn_rois.lod().back();
  }
  int lod_size = fpn_rois_lod.size() - 1;
  int roi_num = fpn_rois_lod[lod_size];

  // get batch id by lod in CPU
  DenseTensor roi_batch_id_list;
  roi_batch_id_list.Resize({roi_num});
  int* roi_batch_id_data = dev_ctx.template HostAlloc<int>(&roi_batch_id_list);
  for (int n = 0; n < lod_size; ++n) {
    for (size_t i = fpn_rois_lod[n]; i < fpn_rois_lod[n + 1]; ++i) {
      roi_batch_id_data[i] = n;
    }
  }
  // copy batch id list to GPU
  DenseTensor roi_batch_id_list_gpu;
  Copy(dev_ctx,
       roi_batch_id_list,
       dev_ctx.GetPlace(),
       true,
       &roi_batch_id_list_gpu);

  DenseTensor sub_lod_list;
  sub_lod_list.Resize({num_level, lod_size});
  int* sub_lod_list_data = dev_ctx.template Alloc<int>(&sub_lod_list);
  phi::funcs::SetConstant<phi::GPUContext, int> set_zero;
  set_zero(dev_ctx, &sub_lod_list, static_cast<int>(0));

  DenseTensor target_lvls;
  target_lvls.Resize({roi_num});
  int* target_lvls_data = dev_ctx.template Alloc<int>(&target_lvls);

  int dist_blocks = NumBlocks(roi_num);
  int threads = kNumCUDAThreads;
  // get target levels and sub_lod list
  GPUDistFpnProposalsHelper<T><<<dist_blocks, threads, 0, dev_ctx.stream()>>>(
      roi_num,
      fpn_rois.data<T>(),
      lod_size,
      refer_level,
      refer_scale,
      max_level,
      min_level,
      roi_batch_id_list_gpu.data<int>(),
      sub_lod_list_data,
      target_lvls_data,
      pixel_offset);
  auto place = dev_ctx.GetPlace();

  DenseTensor index_in_t;
  index_in_t.Resize({roi_num});
  int* idx_in = dev_ctx.template Alloc<int>(&index_in_t);
  funcs::ForRange<phi::GPUContext> for_range(dev_ctx, roi_num);
  for_range(funcs::RangeInitFunctor{0, 1, idx_in});

  DenseTensor keys_out_t;
  keys_out_t.Resize({roi_num});
  int* keys_out = dev_ctx.template Alloc<int>(&keys_out_t);
  DenseTensor index_out_t;
  index_out_t.Resize({roi_num});
  int* idx_out = dev_ctx.template Alloc<int>(&index_out_t);

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs<int, int>(nullptr,
                                            temp_storage_bytes,
                                            target_lvls_data,
                                            keys_out,
                                            idx_in,
                                            idx_out,
                                            roi_num,
                                            0,
                                            sizeof(int) * 8,
                                            dev_ctx.stream());
  // Allocate temporary storage
  auto d_temp_storage = phi::memory_utils::Alloc(place, temp_storage_bytes);

  // Run sorting operation
  // sort target level to get corresponding index
  cub::DeviceRadixSort::SortPairs<int, int>(d_temp_storage->ptr(),
                                            temp_storage_bytes,
                                            target_lvls_data,
                                            keys_out,
                                            idx_in,
                                            idx_out,
                                            roi_num,
                                            0,
                                            sizeof(int) * 8,
                                            dev_ctx.stream());

  restore_index->Resize({roi_num, 1});
  int* restore_idx_data = dev_ctx.template Alloc<int>(restore_index);
  // sort current index to get restore index
  cub::DeviceRadixSort::SortPairs<int, int>(d_temp_storage->ptr(),
                                            temp_storage_bytes,
                                            idx_out,
                                            keys_out,
                                            idx_in,
                                            restore_idx_data,
                                            roi_num,
                                            0,
                                            sizeof(int) * 8,
                                            dev_ctx.stream());

  int start = 0;

  std::vector<int> sub_lod_list_cpu(lod_size * num_level);
  memory_utils::Copy(phi::CPUPlace(),
                     sub_lod_list_cpu.data(),
                     place,
                     sub_lod_list_data,
                     sizeof(int) * lod_size * num_level,
                     dev_ctx.stream());
  dev_ctx.Wait();

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
      phi::funcs::GPUGather<T>(dev_ctx, fpn_rois, sub_idx, multi_fpn_rois[i]);
    } else {
      multi_fpn_rois[i]->Resize({sub_rois_num, funcs::kBoxDim});
      dev_ctx.template Alloc<T>(multi_fpn_rois[i]);
    }
    if (multi_level_rois_num.size() > 0) {
      DenseTensor* rois_num_t = multi_level_rois_num[i];
      Copy(dev_ctx, sub_lod, dev_ctx.GetPlace(), true, rois_num_t);
      rois_num_t->Resize({lod_size});
    }
    LegacyLoD lod;
    lod.emplace_back(offset);
    multi_fpn_rois[i]->set_lod(lod);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(distribute_fpn_proposals,
                   GPU,
                   ALL_LAYOUT,
                   phi::DistributeFpnProposalsKernel,
                   float,
                   double) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
