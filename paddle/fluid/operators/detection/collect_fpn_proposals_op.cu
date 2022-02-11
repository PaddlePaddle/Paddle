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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include <paddle/fluid/memory/allocation/allocator.h>
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/fluid/operators/detection/collect_fpn_proposals_op.h"
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static constexpr int kNumCUDAThreads = 64;
static constexpr int kNumMaxinumNumBlocks = 4096;

const int kBBoxSize = 4;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

static __global__ void GetLengthLoD(const int nthreads, const int* batch_ids,
                                    int* length_lod) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    platform::CudaAtomicAdd(length_lod + batch_ids[i], 1);
  }
}

template <typename DeviceContext, typename T>
class GPUCollectFpnProposalsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto roi_ins = ctx.MultiInput<LoDTensor>("MultiLevelRois");
    const auto score_ins = ctx.MultiInput<LoDTensor>("MultiLevelScores");
    auto fpn_rois = ctx.Output<LoDTensor>("FpnRois");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    const int post_nms_topN = ctx.Attr<int>("post_nms_topN");

    // concat inputs along axis = 0
    int roi_offset = 0;
    int score_offset = 0;
    int total_roi_num = 0;
    for (size_t i = 0; i < roi_ins.size(); ++i) {
      total_roi_num += roi_ins[i]->dims()[0];
    }

    int real_post_num = min(post_nms_topN, total_roi_num);
    fpn_rois->mutable_data<T>({real_post_num, kBBoxSize}, dev_ctx.GetPlace());
    Tensor concat_rois;
    Tensor concat_scores;
    T* concat_rois_data = concat_rois.mutable_data<T>(
        {total_roi_num, kBBoxSize}, dev_ctx.GetPlace());
    T* concat_scores_data =
        concat_scores.mutable_data<T>({total_roi_num, 1}, dev_ctx.GetPlace());
    Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({total_roi_num});
    int* roi_batch_id_data =
        roi_batch_id_list.mutable_data<int>(platform::CPUPlace());
    int index = 0;
    int lod_size;
    auto place = dev_ctx.GetPlace();

    auto multi_rois_num = ctx.MultiInput<Tensor>("MultiLevelRoIsNum");
    for (size_t i = 0; i < roi_ins.size(); ++i) {
      auto roi_in = roi_ins[i];
      auto score_in = score_ins[i];
      if (multi_rois_num.size() > 0) {
        framework::Tensor temp;
        paddle::framework::TensorCopySync(*multi_rois_num[i],
                                          platform::CPUPlace(), &temp);
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

      memory::Copy(place, concat_rois_data + roi_offset, place,
                   roi_in->data<T>(), roi_in->numel() * sizeof(T),
                   dev_ctx.stream());
      memory::Copy(place, concat_scores_data + score_offset, place,
                   score_in->data<T>(), score_in->numel() * sizeof(T),
                   dev_ctx.stream());
      roi_offset += roi_in->numel();
      score_offset += score_in->numel();
    }

    // copy batch id list to GPU
    Tensor roi_batch_id_list_gpu;
    framework::TensorCopy(roi_batch_id_list, dev_ctx.GetPlace(),
                          &roi_batch_id_list_gpu);

    Tensor index_in_t;
    int* idx_in =
        index_in_t.mutable_data<int>({total_roi_num}, dev_ctx.GetPlace());
    platform::ForRange<platform::CUDADeviceContext> for_range_total(
        dev_ctx, total_roi_num);
    for_range_total(RangeInitFunctor{0, 1, idx_in});

    Tensor keys_out_t;
    T* keys_out =
        keys_out_t.mutable_data<T>({total_roi_num}, dev_ctx.GetPlace());
    Tensor index_out_t;
    int* idx_out =
        index_out_t.mutable_data<int>({total_roi_num}, dev_ctx.GetPlace());

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending<T, int>(
        nullptr, temp_storage_bytes, concat_scores.data<T>(), keys_out, idx_in,
        idx_out, total_roi_num, 0, sizeof(T) * 8, dev_ctx.stream());
    // Allocate temporary storage
    auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);

    // Run sorting operation
    // sort score to get corresponding index
    cub::DeviceRadixSort::SortPairsDescending<T, int>(
        d_temp_storage->ptr(), temp_storage_bytes, concat_scores.data<T>(),
        keys_out, idx_in, idx_out, total_roi_num, 0, sizeof(T) * 8,
        dev_ctx.stream());
    index_out_t.Resize({real_post_num});
    Tensor sorted_rois;
    sorted_rois.mutable_data<T>({real_post_num, kBBoxSize}, dev_ctx.GetPlace());
    Tensor sorted_batch_id;
    sorted_batch_id.mutable_data<int>({real_post_num}, dev_ctx.GetPlace());
    GPUGather<T>(dev_ctx, concat_rois, index_out_t, &sorted_rois);
    GPUGather<int>(dev_ctx, roi_batch_id_list_gpu, index_out_t,
                   &sorted_batch_id);

    Tensor batch_index_t;
    int* batch_idx_in =
        batch_index_t.mutable_data<int>({real_post_num}, dev_ctx.GetPlace());
    platform::ForRange<platform::CUDADeviceContext> for_range_post(
        dev_ctx, real_post_num);
    for_range_post(RangeInitFunctor{0, 1, batch_idx_in});

    Tensor out_id_t;
    int* out_id_data =
        out_id_t.mutable_data<int>({real_post_num}, dev_ctx.GetPlace());
    // Determine temporary device storage requirements
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs<int, int>(
        nullptr, temp_storage_bytes, sorted_batch_id.data<int>(), out_id_data,
        batch_idx_in, index_out_t.data<int>(), real_post_num, 0,
        sizeof(int) * 8, dev_ctx.stream());
    // Allocate temporary storage
    d_temp_storage = memory::Alloc(place, temp_storage_bytes);

    // Run sorting operation
    // sort batch_id to get corresponding index
    cub::DeviceRadixSort::SortPairs<int, int>(
        d_temp_storage->ptr(), temp_storage_bytes, sorted_batch_id.data<int>(),
        out_id_data, batch_idx_in, index_out_t.data<int>(), real_post_num, 0,
        sizeof(int) * 8, dev_ctx.stream());

    GPUGather<T>(dev_ctx, sorted_rois, index_out_t, fpn_rois);

    Tensor length_lod;
    int* length_lod_data =
        length_lod.mutable_data<int>({lod_size}, dev_ctx.GetPlace());
    pten::funcs::SetConstant<platform::CUDADeviceContext, int> set_zero;
    set_zero(dev_ctx, &length_lod, static_cast<int>(0));

    int blocks = NumBlocks(real_post_num);
    int threads = kNumCUDAThreads;

    // get length-based lod by batch ids
    GetLengthLoD<<<blocks, threads, 0, dev_ctx.stream()>>>(
        real_post_num, out_id_data, length_lod_data);
    std::vector<int> length_lod_cpu(lod_size);
    memory::Copy(platform::CPUPlace(), length_lod_cpu.data(), place,
                 length_lod_data, sizeof(int) * lod_size, dev_ctx.stream());
    dev_ctx.Wait();

    std::vector<size_t> offset(1, 0);
    for (int i = 0; i < lod_size; ++i) {
      offset.emplace_back(offset.back() + length_lod_cpu[i]);
    }

    if (ctx.HasOutput("RoisNum")) {
      auto* rois_num = ctx.Output<Tensor>("RoisNum");
      int* rois_num_data = rois_num->mutable_data<int>({lod_size}, place);
      memory::Copy(place, rois_num_data, place, length_lod_data,
                   lod_size * sizeof(int), dev_ctx.stream());
    }

    framework::LoD lod;
    lod.emplace_back(offset);
    fpn_rois->set_lod(lod);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    collect_fpn_proposals,
    ops::GPUCollectFpnProposalsOpKernel<paddle::platform::CUDADeviceContext,
                                        float>,
    ops::GPUCollectFpnProposalsOpKernel<paddle::platform::CUDADeviceContext,
                                        double>);
