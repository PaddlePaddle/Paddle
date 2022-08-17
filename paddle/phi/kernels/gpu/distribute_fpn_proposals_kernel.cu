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

#include <algorithm>
#include <vector>

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include <paddle/fluid/memory/allocation/allocator.h>
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

static constexpr int kNumCUDAThreads = 64;
static constexpr int kNumMaxinumNumBlocks = 4096;

int const BBoxSize = 4;

template <typename T>
inline HOSTDEVICE T RoIArea(const T* box, bool pixel_offset = true) {
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

struct RangeInitFunctor {
  int start_;
  int delta_;
  int* out_;
  __device__ void operator()(size_t i) { out_[i] = start_ + i * delta_; }
};

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
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
    T roi_area = RoIArea(offset_roi, pixel_offset);
    T roi_scale = sqrt(roi_area);
    int tgt_lvl = floor(
        log2(roi_scale / static_cast<T>(refer_scale) + (T)1e-8) + refer_level);
    tgt_lvl = min(max_level, max(tgt_lvl, min_level));
    target_lvls[i] = tgt_lvl;
    // compute number of rois in the same batch and same target level
    paddle::platform::CudaAtomicAdd(
        sub_lod_list + (tgt_lvl - min_level) * lod_size + roi_batch_ind, 1);
  }
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
    std::vector<DenseTensor*> multi_level_roisnum) {
  int num_level = max_level - min_level + 1;

  // check that the fpnrois is not empty
  if (!roisnum.initialized()) {
    PADDLE_ENFORCE_EQ(
        fpnrois.lod().size(),
        1UL,
        phi::errors::InvalidArgument("DistributeFpnProposalsOp needs LoD"
                                     "with one level"));
  }

  std::vector<size_t> fpn_rois_lod;
  if (roisnum.initialized()) {
    fpn_rois_lod = GetLodFromRoisNum<Context>(ctx, &roisnum);
  } else {
    fpn_rois_lod = fpnrois.lod().back();
  }
  int lod_size = fpn_rois_lod.size() - 1;
  int roi_num = fpn_rois_lod[lod_size];

  // get batch id by lod in CPU
  DenseTensor roi_batch_id_list;
  roi_batch_id_list.Resize(phi::make_ddim({roi_num}));
  int* roi_batch_id_data = ctx.template Alloc<int>(&roi_batch_id_list);
  for (int n = 0; n < lod_size; ++n) {
    for (size_t i = fpn_rois_lod[n]; i < fpn_rois_lod[n + 1]; ++i) {
      roi_batch_id_data[i] = n;
    }
  }
  // copy batch id list to GPU
  DenseTensor roi_batch_id_list_gpu;
  phi::Copy(
      ctx, roi_batch_id_list, ctx.GetPlace(), true, &roi_batch_id_list_gpu);

  DenseTensor sub_lod_list;
  sub_lod_list.Resize(phi::make_ddim({num_level, lod_size}));
  int* sub_lod_list_data = ctx.template Alloc<int>(&sub_lod_list);
  phi::funcs::SetConstant<Context, int> set_zero;
  set_zero(ctx, &sub_lod_list, static_cast<int>(0));

  DenseTensor target_lvls;
  target_lvls.Resize(phi::make_ddim({roi_num}));
  int* target_lvls_data = ctx.template Alloc<int>(&target_lvls);

  int dist_blocks = NumBlocks(roi_num);
  int threads = kNumCUDAThreads;
  // get target levels and sub_lod list
  GPUDistFpnProposalsHelper<T><<<dist_blocks, threads, 0, ctx.stream()>>>(
      roi_num,
      fpnrois.data<T>(),
      lod_size,
      refer_level,
      refer_scale,
      max_level,
      min_level,
      roi_batch_id_list_gpu.data<int>(),
      sub_lod_list_data,
      target_lvls_data,
      pixel_offset);
  auto place = ctx.GetPlace();

  DenseTensor index_in_t;
  index_in_t.Resize(phi::make_ddim({roi_num}));
  int* idx_in = ctx.template Alloc<int>(&index_in_t);
  phi::funcs::ForRange<Context> for_range(ctx, roi_num);
  for_range(RangeInitFunctor{0, 1, idx_in});

  DenseTensor keys_out_t;
  keys_out_t.Resize(phi::make_ddim({roi_num}));
  int* keys_out = ctx.template Alloc<int>(&keys_out_t);
  DenseTensor index_out_t;
  index_out_t.Resize(phi::make_ddim({roi_num}));
  int* idx_out = ctx.template Alloc<int>(&index_out_t);

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
                                            ctx.stream());
  // Allocate temporary storage
  auto d_temp_storage = paddle::memory::Alloc(place, temp_storage_bytes);

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
                                            ctx.stream());

  restore_index->Resize(phi::make_ddim({roi_num, 1}));
  int* restore_idx_data = ctx.template Alloc<int>(restore_index);
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
                                            ctx.stream());

  int start = 0;
  auto multi_rois_num = multi_fpn_rois;

  std::vector<int> sub_lod_list_cpu(lod_size * num_level);
  paddle::memory::Copy(CPUPlace(),
                       sub_lod_list_cpu.data(),
                       place,
                       sub_lod_list_data,
                       sizeof(int) * lod_size * num_level,
                       ctx.stream());
  ctx.Wait();

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
      multi_fpn_rois[i]->Resize(phi::make_ddim({sub_rois_num, kBoxDim}));
      ctx.template Alloc<T>(multi_fpn_rois[i]);
      phi::funcs::GPUGather<T>(ctx, fpnrois, sub_idx, multi_fpn_rois[i]);
    } else {
      multi_fpn_rois[i]->Resize(phi::make_ddim({sub_rois_num, kBoxDim}));
      ctx.template Alloc<T>(multi_fpn_rois[i]);
    }
    if (multi_rois_num.size() > 0) {
      DenseTensor* rois_num_t = multi_rois_num[i];
      phi::Copy(ctx, sub_lod, ctx.GetPlace(), true, rois_num_t);
      rois_num_t->Resize(phi::make_ddim({lod_size}));
    }
    phi::LoD lod;
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
                   double) {}
