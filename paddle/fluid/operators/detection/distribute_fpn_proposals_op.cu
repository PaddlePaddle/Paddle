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
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/fluid/operators/detection/distribute_fpn_proposals_op.h"
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static constexpr int kNumCUDAThreads = 64;
static constexpr int kNumMaxinumNumBlocks = 4096;

int const BBoxSize = 4;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <class T>
__global__ void GPUDistFpnProposalsHelper(
    const int nthreads, const T* rois, const int lod_size,
    const int refer_level, const int refer_scale, const int max_level,
    const int min_level, int* roi_batch_id_data, int* sub_lod_list,
    int* target_lvls, bool pixel_offset = true) {
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
    platform::CudaAtomicAdd(
        sub_lod_list + (tgt_lvl - min_level) * lod_size + roi_batch_ind, 1);
  }
}

template <typename DeviceContext, typename T>
class GPUDistributeFpnProposalsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* fpn_rois = ctx.Input<paddle::framework::LoDTensor>("FpnRois");

    auto multi_fpn_rois = ctx.MultiOutput<LoDTensor>("MultiFpnRois");
    auto* restore_index = ctx.Output<Tensor>("RestoreIndex");

    const int min_level = ctx.Attr<int>("min_level");
    const int max_level = ctx.Attr<int>("max_level");
    const int refer_level = ctx.Attr<int>("refer_level");
    const int refer_scale = ctx.Attr<int>("refer_scale");
    const bool pixel_offset = ctx.Attr<bool>("pixel_offset");
    int num_level = max_level - min_level + 1;

    // check that the fpn_rois is not empty
    if (!ctx.HasInput("RoisNum")) {
      PADDLE_ENFORCE_EQ(
          fpn_rois->lod().size(), 1UL,
          platform::errors::InvalidArgument("DistributeFpnProposalsOp needs LoD"
                                            "with one level"));
    }

    std::vector<size_t> fpn_rois_lod;
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num = ctx.Input<Tensor>("RoisNum");
      fpn_rois_lod = GetLodFromRoisNum(rois_num);
    } else {
      fpn_rois_lod = fpn_rois->lod().back();
    }
    int lod_size = fpn_rois_lod.size() - 1;
    int roi_num = fpn_rois_lod[lod_size];

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    // get batch id by lod in CPU
    Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({roi_num});
    int* roi_batch_id_data =
        roi_batch_id_list.mutable_data<int>(platform::CPUPlace());
    for (int n = 0; n < lod_size; ++n) {
      for (size_t i = fpn_rois_lod[n]; i < fpn_rois_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
    // copy batch id list to GPU
    Tensor roi_batch_id_list_gpu;
    framework::TensorCopySync(roi_batch_id_list, dev_ctx.GetPlace(),
                              &roi_batch_id_list_gpu);

    Tensor sub_lod_list;
    sub_lod_list.Resize({num_level, lod_size});
    int* sub_lod_list_data = sub_lod_list.mutable_data<int>(dev_ctx.GetPlace());
    math::SetConstant<platform::CUDADeviceContext, int> set_zero;
    set_zero(dev_ctx, &sub_lod_list, static_cast<int>(0));

    Tensor target_lvls;
    target_lvls.Resize({roi_num});
    int* target_lvls_data = target_lvls.mutable_data<int>(dev_ctx.GetPlace());

    int dist_blocks = NumBlocks(roi_num);
    int threads = kNumCUDAThreads;
    // get target levels and sub_lod list
    GPUDistFpnProposalsHelper<T><<<dist_blocks, threads>>>(
        roi_num, fpn_rois->data<T>(), lod_size, refer_level, refer_scale,
        max_level, min_level, roi_batch_id_list_gpu.data<int>(),
        sub_lod_list_data, target_lvls_data, pixel_offset);
    dev_ctx.Wait();
    auto place = BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace());

    Tensor index_in_t;
    int* idx_in = index_in_t.mutable_data<int>({roi_num}, dev_ctx.GetPlace());
    platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx, roi_num);
    for_range(RangeInitFunctor{0, 1, idx_in});

    Tensor keys_out_t;
    int* keys_out = keys_out_t.mutable_data<int>({roi_num}, dev_ctx.GetPlace());
    Tensor index_out_t;
    int* idx_out = index_out_t.mutable_data<int>({roi_num}, dev_ctx.GetPlace());

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs<int, int>(nullptr, temp_storage_bytes,
                                              target_lvls_data, keys_out,
                                              idx_in, idx_out, roi_num);
    // Allocate temporary storage
    auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);

    // Run sorting operation
    // sort target level to get corresponding index
    cub::DeviceRadixSort::SortPairs<int, int>(
        d_temp_storage->ptr(), temp_storage_bytes, target_lvls_data, keys_out,
        idx_in, idx_out, roi_num);

    int* restore_idx_data =
        restore_index->mutable_data<int>({roi_num, 1}, dev_ctx.GetPlace());
    // sort current index to get restore index
    cub::DeviceRadixSort::SortPairs<int, int>(
        d_temp_storage->ptr(), temp_storage_bytes, idx_out, keys_out, idx_in,
        restore_idx_data, roi_num);

    int start = 0;
    auto multi_rois_num = ctx.MultiOutput<Tensor>("MultiLevelRoIsNum");

    for (int i = 0; i < num_level; ++i) {
      Tensor sub_lod = sub_lod_list.Slice(i, i + 1);
      int* sub_lod_data = sub_lod.data<int>();
      // transfer length-based lod to offset-based lod
      std::vector<size_t> offset(1, 0);
      std::vector<int> sub_lod_cpu(lod_size);
      memory::Copy(platform::CPUPlace(), sub_lod_cpu.data(), place,
                   sub_lod_data, sizeof(int) * lod_size, dev_ctx.stream());
      dev_ctx.Wait();
      for (int j = 0; j < lod_size; ++j) {
        offset.emplace_back(offset.back() + sub_lod_cpu[j]);
      }

      int sub_rois_num = offset.back();

      int end = start + sub_rois_num;
      if (end > start) {
        Tensor sub_idx = index_out_t.Slice(start, end);
        start = end;
        multi_fpn_rois[i]->mutable_data<T>({sub_rois_num, kBoxDim},
                                           dev_ctx.GetPlace());
        GPUGather<T>(dev_ctx, *fpn_rois, sub_idx, multi_fpn_rois[i]);
      } else {
        multi_fpn_rois[i]->mutable_data<T>({sub_rois_num, kBoxDim},
                                           dev_ctx.GetPlace());
      }
      if (multi_rois_num.size() > 0) {
        Tensor* rois_num_t = multi_rois_num[i];
        TensorCopySync(sub_lod, dev_ctx.GetPlace(), rois_num_t);
        rois_num_t->Resize({lod_size});
      }
      framework::LoD lod;
      lod.emplace_back(offset);
      multi_fpn_rois[i]->set_lod(lod);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    distribute_fpn_proposals,
    ops::GPUDistributeFpnProposalsOpKernel<paddle::platform::CUDADeviceContext,
                                           float>,
    ops::GPUDistributeFpnProposalsOpKernel<paddle::platform::CUDADeviceContext,
                                           double>);
