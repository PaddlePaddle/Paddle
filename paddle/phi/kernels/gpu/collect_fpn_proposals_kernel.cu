// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/detection/bbox_util.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/strided_memcpy.h"
#include "paddle/phi/kernels/impl/collect_fpn_proposals_kernel_impl.h"
#include "paddle/utils/optional.h"

namespace phi {

static constexpr int kNumCUDAThreads = 64;
static constexpr int kNumMaximumNumBlocks = 4096;

const int kBBoxSize = 4;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

static __global__ void GetLengthLoD(const int nthreads,
                                    const int* batch_ids,
                                    int* length_lod) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    phi::CudaAtomicAdd(length_lod + batch_ids[i], 1);
  }
}

template <typename T, typename Context>
void GPUCollectFpnProposalsOpKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& multi_level_rois,
    const std::vector<const DenseTensor*>& multi_level_scores,
    const paddle::optional<std::vector<const DenseTensor*>>&
        multi_level_rois_num,
    int post_nms_topn,
    DenseTensor* fpn_rois_out,
    DenseTensor* rois_num_out) {
  const auto roi_ins = multi_level_rois;
  const auto score_ins = multi_level_scores;
  auto fpn_rois = fpn_rois_out;

  const int post_nms_topN = post_nms_topn;

  // concat inputs along axis = 0
  int roi_offset = 0;
  int score_offset = 0;
  int total_roi_num = 0;
  for (size_t i = 0; i < roi_ins.size(); ++i) {
    total_roi_num += roi_ins[i]->dims()[0];
  }

  int real_post_num = min(post_nms_topN, total_roi_num);
  fpn_rois->Resize({real_post_num, kBBoxSize});
  dev_ctx.template Alloc<T>(fpn_rois);
  phi::DenseTensor concat_rois;
  phi::DenseTensor concat_scores;
  concat_rois.Resize({total_roi_num, kBBoxSize});
  T* concat_rois_data = dev_ctx.template Alloc<T>(&concat_rois);
  concat_scores.Resize({total_roi_num, 1});
  T* concat_scores_data = dev_ctx.template Alloc<T>(&concat_scores);
  phi::DenseTensor roi_batch_id_list;
  roi_batch_id_list.Resize({total_roi_num});
  int* roi_batch_id_data = dev_ctx.template HostAlloc<int>(&roi_batch_id_list);
  int index = 0;
  int lod_size;
  auto place = dev_ctx.GetPlace();

  auto multi_rois_num = multi_level_rois_num
                            ? multi_level_rois_num.get()
                            : std::vector<const DenseTensor*>();
  for (size_t i = 0; i < roi_ins.size(); ++i) {
    auto roi_in = roi_ins[i];
    auto score_in = score_ins[i];
    if (multi_rois_num.size() > 0) {
      phi::DenseTensor temp;
      phi::Copy(dev_ctx, *multi_rois_num[i], phi::CPUPlace(), true, &temp);
      const int* length_in = temp.data<int>();
      lod_size = multi_rois_num[i]->numel();
      for (size_t n = 0; n < lod_size; ++n) {
        for (size_t j = 0; j < length_in[n]; ++j) {
          roi_batch_id_data[index++] = n;
        }
      }
    } else {
      auto length_in = roi_in->lod().back();
      lod_size = length_in.size() - 1;
      for (size_t n = 0; n < lod_size; ++n) {
        for (size_t j = length_in[n]; j < length_in[n + 1]; ++j) {
          roi_batch_id_data[index++] = n;
        }
      }
    }

    phi::memory_utils::Copy(place,
                            concat_rois_data + roi_offset,
                            place,
                            roi_in->data<T>(),
                            roi_in->numel() * sizeof(T),
                            dev_ctx.stream());
    phi::memory_utils::Copy(place,
                            concat_scores_data + score_offset,
                            place,
                            score_in->data<T>(),
                            score_in->numel() * sizeof(T),
                            dev_ctx.stream());
    roi_offset += roi_in->numel();
    score_offset += score_in->numel();
  }

  // copy batch id list to GPU
  phi::DenseTensor roi_batch_id_list_gpu;
  phi::Copy(dev_ctx,
            roi_batch_id_list,
            dev_ctx.GetPlace(),
            false,
            &roi_batch_id_list_gpu);

  phi::DenseTensor index_in_t;
  index_in_t.Resize({total_roi_num});
  int* idx_in = dev_ctx.template Alloc<int>(&index_in_t);
  phi::funcs::ForRange<phi::GPUContext> for_range_total(dev_ctx, total_roi_num);
  for_range_total(phi::funcs::RangeInitFunctor{0, 1, idx_in});

  phi::DenseTensor keys_out_t;
  keys_out_t.Resize({total_roi_num});
  T* keys_out = dev_ctx.template Alloc<T>(&keys_out_t);
  phi::DenseTensor index_out_t;
  index_out_t.Resize({total_roi_num});
  int* idx_out = dev_ctx.template Alloc<int>(&index_out_t);

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending<T, int>(nullptr,
                                                    temp_storage_bytes,
                                                    concat_scores.data<T>(),
                                                    keys_out,
                                                    idx_in,
                                                    idx_out,
                                                    total_roi_num,
                                                    0,
                                                    sizeof(T) * 8,
                                                    dev_ctx.stream());
  // Allocate temporary storage
  auto d_temp_storage = phi::memory_utils::Alloc(place, temp_storage_bytes);

  // Run sorting operation
  // sort score to get corresponding index
  cub::DeviceRadixSort::SortPairsDescending<T, int>(d_temp_storage->ptr(),
                                                    temp_storage_bytes,
                                                    concat_scores.data<T>(),
                                                    keys_out,
                                                    idx_in,
                                                    idx_out,
                                                    total_roi_num,
                                                    0,
                                                    sizeof(T) * 8,
                                                    dev_ctx.stream());
  index_out_t.Resize({real_post_num});
  phi::DenseTensor sorted_rois;
  sorted_rois.Resize({real_post_num, kBBoxSize});
  dev_ctx.template Alloc<T>(&sorted_rois);
  phi::DenseTensor sorted_batch_id;
  sorted_batch_id.Resize({real_post_num});
  dev_ctx.template Alloc<int>(&sorted_batch_id);
  phi::funcs::GPUGather<T>(dev_ctx, concat_rois, index_out_t, &sorted_rois);
  phi::funcs::GPUGather<int>(
      dev_ctx, roi_batch_id_list_gpu, index_out_t, &sorted_batch_id);

  phi::DenseTensor batch_index_t;
  batch_index_t.Resize({real_post_num});
  int* batch_idx_in = dev_ctx.template Alloc<int>(&batch_index_t);
  phi::funcs::ForRange<phi::GPUContext> for_range_post(dev_ctx, real_post_num);
  for_range_post(phi::funcs::RangeInitFunctor{0, 1, batch_idx_in});

  phi::DenseTensor out_id_t;
  out_id_t.Resize({real_post_num});
  int* out_id_data = dev_ctx.template Alloc<int>(&out_id_t);
  // Determine temporary device storage requirements
  temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs<int, int>(nullptr,
                                            temp_storage_bytes,
                                            sorted_batch_id.data<int>(),
                                            out_id_data,
                                            batch_idx_in,
                                            index_out_t.data<int>(),
                                            real_post_num,
                                            0,
                                            sizeof(int) * 8,
                                            dev_ctx.stream());
  // Allocate temporary storage
  d_temp_storage = phi::memory_utils::Alloc(place, temp_storage_bytes);

  // Run sorting operation
  // sort batch_id to get corresponding index
  cub::DeviceRadixSort::SortPairs<int, int>(d_temp_storage->ptr(),
                                            temp_storage_bytes,
                                            sorted_batch_id.data<int>(),
                                            out_id_data,
                                            batch_idx_in,
                                            index_out_t.data<int>(),
                                            real_post_num,
                                            0,
                                            sizeof(int) * 8,
                                            dev_ctx.stream());

  phi::funcs::GPUGather<T>(dev_ctx, sorted_rois, index_out_t, fpn_rois);

  phi::DenseTensor length_lod;
  length_lod.Resize({lod_size});
  int* length_lod_data = dev_ctx.template Alloc<int>(&length_lod);
  phi::funcs::SetConstant<phi::GPUContext, int> set_zero;
  set_zero(dev_ctx, &length_lod, static_cast<int>(0));

  int blocks = NumBlocks(real_post_num);
  int threads = kNumCUDAThreads;

  // get length-based lod by batch ids
  GetLengthLoD<<<blocks, threads, 0, dev_ctx.stream()>>>(
      real_post_num, out_id_data, length_lod_data);
  std::vector<int> length_lod_cpu(lod_size);
  phi::memory_utils::Copy(phi::CPUPlace(),
                          length_lod_cpu.data(),
                          place,
                          length_lod_data,
                          sizeof(int) * lod_size,
                          dev_ctx.stream());
  dev_ctx.Wait();

  std::vector<size_t> offset(1, 0);
  for (int i = 0; i < lod_size; ++i) {
    offset.emplace_back(offset.back() + length_lod_cpu[i]);
  }

  if (rois_num_out != nullptr) {
    auto* rois_num = rois_num_out;
    rois_num->Resize({lod_size});
    int* rois_num_data = dev_ctx.template Alloc<int>(rois_num);
    phi::memory_utils::Copy(place,
                            rois_num_data,
                            place,
                            length_lod_data,
                            lod_size * sizeof(int),
                            dev_ctx.stream());
  }

  phi::LegacyLoD lod;
  lod.emplace_back(offset);
  fpn_rois->set_lod(lod);
}
}  // namespace phi

PD_REGISTER_KERNEL(collect_fpn_proposals,
                   GPU,
                   ALL_LAYOUT,
                   phi::GPUCollectFpnProposalsOpKernel,
                   float,
                   double) {
  kernel->InputAt(2).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
}
