/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <limits>

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_launch_config.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fused_token_prune_op.cu.h"

namespace paddle {
namespace operators {

template <typename T>
struct AttnMaskFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return b >= 0 ? a : 0;
  }
};

__global__ void FillIndex(int64_t* indices, int num_raws, int num_cols) {
  int num_threads = num_raws * num_cols;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (; tid < num_threads; tid += stride) {
    int col = tid % num_cols;
    indices[tid] = (int64_t)col;
  }
}

template <typename T>
__global__ void TakeAlongAxis(const T* src,
                              T* dst,
                              int64_t* indices,
                              int num_raws,
                              int src_num_cols,
                              int dst_num_cols,
                              int num_elements) {
  int num_threads = num_raws * dst_num_cols;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (; tid < num_threads; tid += stride) {
    int raw = tid / dst_num_cols;
    int col = tid % dst_num_cols;
    for (int i = 0; i < num_elements; ++i) {
      dst[tid * num_elements + i] =
          *(src + (raw * src_num_cols + indices[tid]) * num_elements + i);
    }
  }
}

template <typename T>
__global__ void MaximumFirst(T* mat, int num_raws, int num_cols, T max_value) {
  int num_threads = num_raws;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; tid < num_threads; tid += stride) {
    mat[tid * num_cols] = max_value;
  }
}

template <typename T>
class FusedTokenPruneOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx = context.cuda_device_context();
    // Inouts
    const phi::DenseTensor* attn = context.Input<phi::DenseTensor>("Attn");
    const phi::DenseTensor* x = context.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* mask = context.Input<phi::DenseTensor>("Mask");
    const phi::DenseTensor* new_mask =
        context.Input<phi::DenseTensor>("NewMask");

    // Input dims
    auto attn_dims = attn->dims();
    auto x_dims = x->dims();
    auto new_mask_dims = new_mask->dims();

    auto bsz = attn_dims[0];
    auto num_heads = attn_dims[1];
    auto max_seq_len = attn_dims[2];
    auto c = x_dims[2];
    int slimmed_x_len = new_mask_dims[2];

    // Attrs
    const bool keep_first_token = context.Attr<bool>("keep_first_token");
    const bool keep_order = context.Attr<bool>("keep_order");

    // Outputs
    phi::DenseTensor* out_slimmed_x =
        context.Output<phi::DenseTensor>("SlimmedX");
    phi::DenseTensor* slimmed_indices =
        context.Output<phi::DenseTensor>("CLSInds");
    auto* out_slimmed_x_data =
        out_slimmed_x->mutable_data<T>(context.GetPlace());
    auto* slimmed_indices_data =
        slimmed_indices->mutable_data<int64_t>(context.GetPlace());

    // Intermediate variable
    phi::DenseTensor attn_tmp;
    auto* attn_tmp_data =
        attn_tmp.mutable_data<T>(attn_dims, context.GetPlace());
    phi::DenseTensor attn_accu;
    auto* attn_accu_data =
        attn_accu.mutable_data<T>({bsz, max_seq_len}, context.GetPlace());
    phi::DenseTensor attn_accu_indices;
    auto* attn_accu_indices_data = attn_accu_indices.mutable_data<int64_t>(
        {bsz, max_seq_len}, context.GetPlace());
    phi::DenseTensor sort_attn_accu;
    auto* sort_attn_accu_data =
        sort_attn_accu.mutable_data<T>({bsz, max_seq_len}, context.GetPlace());
    phi::DenseTensor sort_attn_accu_indices;
    auto* sort_attn_accu_indices_data =
        sort_attn_accu_indices.mutable_data<int64_t>({bsz, max_seq_len},
                                                     context.GetPlace());
    phi::DenseTensor temp_storage;

    // 1. Filter attn by mask
    std::vector<const phi::DenseTensor*> ins;
    std::vector<phi::DenseTensor*> outs;
    ins.emplace_back(attn);
    ins.emplace_back(mask);
    outs.emplace_back(&attn_tmp);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, -1, AttnMaskFunctor<T>());

    // 2. Reduce sum
    const std::vector<int64_t> reduce_dims{1, 2};
    phi::Reduce<T, kps::AddFunctor, kps::IdentityFunctor>(dev_ctx,
                                                          attn_tmp,
                                                          false,
                                                          reduce_dims,
                                                          false,
                                                          attn_accu.dtype(),
                                                          &attn_accu);
    // 3. Prepare token indices
    phi::backends::gpu::GpuLaunchConfig config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, bsz * max_seq_len);
    FillIndex<<<config.block_per_grid,
                config.thread_per_block,
                0,
                dev_ctx.stream()>>>(attn_accu_indices_data, bsz, max_seq_len);

    // 4. Sort token indices by attn
    if (keep_first_token) {
      T max = std::numeric_limits<T>::max();
      config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, bsz);
      MaximumFirst<T>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(attn_accu_data, bsz, max_seq_len, max);
    }
    size_t temp_storage_bytes = -1;
    int num_items = bsz * max_seq_len;
    int num_segments = bsz;

    cub::CountingInputIterator<int64_t> counting_iter(0);
    cub::TransformInputIterator<int64_t,
                                SegmentOffsetIter,
                                cub::CountingInputIterator<int64_t>>
        segment_offsets_t(counting_iter, SegmentOffsetIter(max_seq_len));
    // Determine temporary device storage requirements
    PADDLE_ENFORCE_GPU_SUCCESS(
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            nullptr,
            temp_storage_bytes,
            attn_accu_data,
            sort_attn_accu_data,
            attn_accu_indices_data,
            sort_attn_accu_indices_data,
            num_items,
            num_segments,
            segment_offsets_t,
            segment_offsets_t + 1,
            0,
            sizeof(T) * 8,
            dev_ctx.stream()));
    // Allocate temporary storage
    int64_t temp_size = temp_storage_bytes;
    auto* temp_storage_data =
        temp_storage.mutable_data<uint8_t>({temp_size}, context.GetPlace());
    // Run sorting operation
    PADDLE_ENFORCE_GPU_SUCCESS(
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp_storage_data,
            temp_storage_bytes,
            attn_accu_data,
            sort_attn_accu_data,
            attn_accu_indices_data,
            sort_attn_accu_indices_data,
            num_items,
            num_segments,
            segment_offsets_t,
            segment_offsets_t + 1,
            0,
            sizeof(T) * 8,
            dev_ctx.stream()));
    // 5. Slice
    auto slimmed_indices_tmp =
        phi::funcs::Slice<int64_t>(dev_ctx,
                                   sort_attn_accu_indices,
                                   {1} /*axes*/,
                                   {0} /*starts*/,
                                   {slimmed_x_len} /*ends*/);
    if (keep_order) {
      // 6. reorder
      num_items = bsz * slimmed_x_len;
      temp_storage_bytes = -1;
      cub::TransformInputIterator<int64_t,
                                  SegmentOffsetIter,
                                  cub::CountingInputIterator<int64_t>>
          segment_offsets_t2(counting_iter, SegmentOffsetIter(slimmed_x_len));
      PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceSegmentedRadixSort::SortKeys(
          nullptr,
          temp_storage_bytes,
          static_cast<int64_t*>(slimmed_indices_tmp.data()),
          static_cast<int64_t*>(slimmed_indices->data()),
          num_items,
          num_segments,
          segment_offsets_t2,
          segment_offsets_t2 + 1,
          0,
          sizeof(int64_t) * 8,
          dev_ctx.stream()));
      temp_size = temp_storage_bytes;
      temp_storage.Resize({temp_size});
      temp_storage_data =
          temp_storage.mutable_data<uint8_t>(context.GetPlace());
      PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceSegmentedRadixSort::SortKeys(
          temp_storage_data,
          temp_storage_bytes,
          static_cast<int64_t*>(slimmed_indices_tmp.data()),
          static_cast<int64_t*>(slimmed_indices->data()),
          num_items,
          num_segments,
          segment_offsets_t2,
          segment_offsets_t2 + 1,
          0,
          sizeof(int64_t) * 8,
          dev_ctx.stream()));
    } else {
      framework::TensorCopy(
          slimmed_indices_tmp, context.GetPlace(), slimmed_indices);
    }
    // 7. Get slimmed X by indices
    config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, bsz * slimmed_x_len);
    TakeAlongAxis<T><<<config.block_per_grid,
                       config.thread_per_block,
                       0,
                       dev_ctx.stream()>>>(x->data<T>(),
                                           out_slimmed_x_data,
                                           slimmed_indices->data<int64_t>(),
                                           bsz,
                                           max_seq_len,
                                           slimmed_x_len,
                                           c);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_token_prune,
                        ops::FusedTokenPruneOpCUDAKernel<float>,
                        ops::FusedTokenPruneOpCUDAKernel<double>);
