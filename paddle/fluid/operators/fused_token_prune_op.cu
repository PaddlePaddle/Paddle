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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fused_token_prune_op.cu.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
struct AttnMaskFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return b >= 0 ? a : 0;
  }
};

__global__ void FillIndex(int64_t* indices, int num_raws, int num_cols) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_raws * num_cols) return;

  int col = tid % num_cols;

  indices[tid] = (int64_t)col;
}

template <typename T>
__global__ void Argsort(T* data, int64_t* indices, int num_raws,
                              int num_cols) {
  auto raw = blockIdx.x * blockDim.x + threadIdx.x;
  if (raw >= num_raws) return;
  thrust::sort_by_key(thrust::seq, data + raw * num_cols,
                      data + (raw + 1) * num_cols, indices + raw * num_cols,
                      thrust::greater<T>());
}

template <typename T>
__global__ void ArgsortKeepFirst(T* data, int64_t* indices, int num_raws,
                              int num_cols) {
  auto raw = blockIdx.x * blockDim.x + threadIdx.x;
  if (raw >= num_raws) return;
  thrust::sort_by_key(thrust::seq, data + raw * num_cols + 1,
                      data + (raw + 1) * num_cols, indices + raw * num_cols + 1,
                      thrust::greater<T>());
}

__global__ void Sort(int64_t* indices, int num_raws, int num_cols) {
  auto raw = blockIdx.x * blockDim.x + threadIdx.x;
  if (raw >= num_raws) return;
  thrust::sort(thrust::seq, indices + raw * num_cols,
    indices + (raw + 1) * num_cols);
}

template <typename T>
__global__ void TakeAlongAxis(const T* src, T* dst, int64_t* indices, int num_raws,
                              int src_num_cols, int dst_num_cols,
                              int num_elements) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_raws * dst_num_cols) return;

  int raw = tid / dst_num_cols;
  int col = tid % dst_num_cols;
  for (int i = 0; i < num_elements; ++i) {
    dst[tid * num_elements + i] =
        *(src + (raw * src_num_cols + (int)indices[tid]) * num_elements + i);
  }
}

template <typename T>
class FusedTokenPruneOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    //Inouts
    const Tensor* attn = context.Input<Tensor>("Attn");
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* mask = context.Input<Tensor>("Mask");
    const Tensor* new_mask = context.Input<Tensor>("NewMask");
    Tensor* out_slimmed_x = context.Output<Tensor>("SlimmedX");
    Tensor* slimmed_indices = context.Output<Tensor>("CLSInds");


    //Attr
    const bool keep_first_token = context.Attr<bool>("keep_first_token");
    const bool keep_order = context.Attr<bool>("keep_order");

    auto* out_slimmed_x_data =
        out_slimmed_x->mutable_data<T>(context.GetPlace());

    Tensor attn_tmp;
    auto attn_dims = attn->dims();
    attn_tmp.Resize(attn_dims);
    auto* attn_tmp_data = attn_tmp.mutable_data<T>(context.GetPlace());

    std::vector<const Tensor*> ins;
    std::vector<Tensor*> outs;
    ins.emplace_back(attn);
    ins.emplace_back(mask);
    outs.emplace_back(&attn_tmp);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        context.cuda_device_context(), ins, &outs, -1, AttnMaskFunctor<T>());

    Tensor attn_accu;
    attn_accu.Resize({attn_dims[0], attn_dims[3]});
    auto* attn_accu_data = attn_accu.mutable_data<T>(context.GetPlace());
    const std::vector<int64_t> reduce_dims{1, 2};
    phi::Reduce<T, kps::AddFunctor, kps::IdentityFunctor>(
        context.cuda_device_context(), attn_tmp, false, reduce_dims, false,
        attn_accu.dtype(), &attn_accu);

    Tensor attn_accu_indices;
    attn_accu_indices.Resize(attn_accu.dims());
    auto* attn_accu_indices_data =
        attn_accu_indices.mutable_data<int64_t>(context.GetPlace());

    int grid_size = attn_dims[0], block_size = ComputeBlockSize(attn_dims[3]); //TODO: N matbe larger than 1024
    FillIndex<<<grid_size, block_size, 0,
                context.cuda_device_context().stream()>>>(
        attn_accu_indices_data, attn_dims[0], attn_dims[3]);

    if (keep_first_token) {
      // ArgsortKeepFirst<
      //   T><<<grid_size, 1, 0, context.cuda_device_context().stream()>>>(
      //         attn_accu_data, attn_accu_indices_data, attn_dims[0], attn_dims[3]);
      for (int raw = 0; raw < attn_dims[0]; ++raw) {
        thrust::sort_by_key(thrust::device, attn_accu_data + raw * attn_dims[3] + 1,
          attn_accu_data + (raw + 1) * attn_dims[3], attn_accu_indices_data + raw * attn_dims[3] + 1,
          thrust::greater<T>());
      }
    } else {
      // Argsort<
      //   T><<<grid_size, 1, 0, context.cuda_device_context().stream()>>>(
      //         attn_accu_data, attn_accu_indices_data, attn_dims[0], attn_dims[3]);
      for (int raw = 0; raw < attn_dims[0]; ++raw) {
        thrust::sort_by_key(thrust::device, attn_accu_data + raw * attn_dims[3] ,
          attn_accu_data + (raw + 1) * attn_dims[3], attn_accu_indices_data + raw * attn_dims[3],
          thrust::greater<T>());
      }
    }

    auto new_mask_dims = new_mask->dims();
    int slimmed_x_len = new_mask_dims[2];
    auto slimmed_indices_tmp =
        phi::funcs::Slice<int64_t>(context.cuda_device_context(), attn_accu_indices,
                               {1} /*axes*/, {0}/*starts*/, {slimmed_x_len}/*ends*/);

    framework::TensorCopy(slimmed_indices_tmp, context.GetPlace(), slimmed_indices);
    
    if (keep_order) {
      // Sort<<<grid_size, 1, 0, context.cuda_device_context().stream()>>>(
      //   slimmed_indices->data<int>(), attn_dims[0], slimmed_x_len);
      for (int raw = 0; raw < attn_dims[0]; ++raw) {
        thrust::sort(thrust::device, slimmed_indices->data<int64_t>() + raw * slimmed_x_len ,
        slimmed_indices->data<int64_t>() + (raw + 1) * slimmed_x_len);
      }
    }
    
    auto x_dims = x->dims();
    block_size = ComputeBlockSize(slimmed_x_len);
    TakeAlongAxis<T><<<grid_size, block_size, 0,
                       context.cuda_device_context().stream()>>>(
        x->data<T>(), out_slimmed_x_data, slimmed_indices->data<int64_t>(),
        attn_dims[0], attn_dims[3], slimmed_x_len, x_dims[2]);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_token_prune,
                        ops::FusedTokenPruneOpCUDAKernel<float>,
                        ops::FusedTokenPruneOpCUDAKernel<double>);
