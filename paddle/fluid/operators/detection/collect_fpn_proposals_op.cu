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

#include <paddle/fluid/memory/allocation/allocator.h>
#include "cub/cub.cuh"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/detection/collect_fpn_proposals_op.h"
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

const int BBoxSize = 4;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

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

static __global__ void GetLengthLoD(const int nthreads, const int* batch_ids,
                                    int* length_lod) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    length_lod[threadIdx.x] = 0;
    __syncthreads();
    platform::CudaAtomicAdd(length_lod + batch_ids[i], 1);
  }
}

template <typename DeviceContext, typename T>
class GPUCollectFpnProposalsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto roi_ins = ctx.MultiInput<LoDTensor>("MultiLayerRois");
    const auto score_ins = ctx.MultiInput<LoDTensor>("MultiLayerScores");
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

    int real_post_num = post_nms_topN;
    if (total_roi_num < post_nms_topN) {
      real_post_num = total_roi_num;
    }
    fpn_rois->mutable_data<T>({real_post_num, BBoxSize}, dev_ctx.GetPlace());
    Tensor concat_rois;
    Tensor concat_scores;
    concat_rois.mutable_data<T>({total_roi_num, BBoxSize}, dev_ctx.GetPlace());
    concat_scores.mutable_data<T>({total_roi_num, 1}, dev_ctx.GetPlace());
    Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({total_roi_num});
    int* roi_batch_id_data =
        roi_batch_id_list.mutable_data<int>(platform::CPUPlace());
    int index = 0;
    int lod_size;
    for (size_t i = 0; i < roi_ins.size(); ++i) {
      auto* roi_in = roi_ins[i];
      auto* score_in = score_ins[i];
      auto roi_lod = roi_in->lod().back();
      lod_size = roi_lod.size() - 1;
      for (size_t n = 0; n < lod_size; ++n) {
        for (size_t j = roi_lod[n]; j < roi_lod[n + 1]; ++j) {
          roi_batch_id_data[index] = n;
          index++;
        }
      }
      auto roi_in_stride = framework::stride_numel(roi_in->dims());
      auto roi_out_stride = framework::stride_numel(concat_rois.dims());
      auto score_in_stride = framework::stride_numel(score_in->dims());
      auto score_out_stride = framework::stride_numel(concat_scores.dims());
      StridedNumelCopyWithAxis<T>(
          ctx.device_context(), 0, concat_rois.data<T>() + roi_offset,
          roi_out_stride, roi_in->data<T>(), roi_in_stride, roi_in_stride[0]);
      StridedNumelCopyWithAxis<T>(ctx.device_context(), 0,
                                  concat_scores.data<T>() + score_offset,
                                  score_out_stride, score_in->data<T>(),
                                  score_in_stride, score_in_stride[0]);
      roi_offset += roi_in_stride[0];
      score_offset += score_in_stride[0];
    }

    // copy batch id list to GPU
    Tensor roi_batch_id_list_gpu;
    framework::TensorCopySync(roi_batch_id_list, dev_ctx.GetPlace(),
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
        idx_out, total_roi_num);
    // Allocate temporary storage
    auto place = boost::get<platform::CUDAPlace>(dev_ctx.GetPlace());
    auto d_temp_storage = memory::Alloc(place, temp_storage_bytes,
                                        memory::Allocator::kScratchpad);

    // Run sorting operation
    // sort score to get corresponding index
    cub::DeviceRadixSort::SortPairsDescending<T, int>(
        d_temp_storage->ptr(), temp_storage_bytes, concat_scores.data<T>(),
        keys_out, idx_in, idx_out, total_roi_num);
    index_out_t.Resize({real_post_num});
    Tensor sorted_rois;
    sorted_rois.mutable_data<T>({real_post_num, BBoxSize}, dev_ctx.GetPlace());
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
        batch_idx_in, index_out_t.data<int>(), real_post_num);
    // Allocate temporary storage
    d_temp_storage = memory::Alloc(place, temp_storage_bytes,
                                   memory::Allocator::kScratchpad);

    // Run sorting operation
    // sort batch_id to get corresponding index
    cub::DeviceRadixSort::SortPairs<int, int>(
        d_temp_storage->ptr(), temp_storage_bytes, sorted_batch_id.data<int>(),
        out_id_data, batch_idx_in, index_out_t.data<int>(), real_post_num);

    GPUGather<T>(dev_ctx, sorted_rois, index_out_t, fpn_rois);

    Tensor length_lod;
    int* length_lod_data =
        length_lod.mutable_data<int>({lod_size}, dev_ctx.GetPlace());
    int blocks = NumBlocks(real_post_num);
    int threads = kNumCUDAThreads;

    // get length-based lod by batch ids
    GetLengthLoD<<<blocks, threads>>>(real_post_num, out_id_data,
                                      length_lod_data);
    std::vector<int> length_lod_cpu(lod_size);
    memory::Copy(platform::CPUPlace(), length_lod_cpu.data(), place,
                 length_lod_data, sizeof(int) * lod_size, 0);

    std::vector<size_t> offset(1, 0);
    for (int i = 0; i < lod_size; ++i) {
      offset.emplace_back(offset.back() + length_lod_cpu[i]);
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
