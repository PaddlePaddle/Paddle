/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include "paddle/fluid/operators/mask_extract_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
struct ge_zero
{
  __host__ __device__ bool operator()(const T &x) const {return x >= 0;}
};

template <typename T>
__global__ void MaskExtractGradKernel(
           const T *d_out_data, 
           const int64_t *offset_data,
           int64_t feat_num,
           int64_t out_len,
           T* d_x_data
           ) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_len) {
      for (int64_t i = 0; i < feat_num; ++i) {
          d_x_data[offset_data[idx] * feat_num + i] = d_out_data[idx * feat_num + i];
      }        
    }
}

template <typename DeviceContext, typename T>
class MaskExtractGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* mask = ctx.Input<framework::LoDTensor>("Mask");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    auto* ids = ctx.Output<framework::LoDTensor>("Ids");
    auto* offset = ctx.Output<framework::LoDTensor>("Offset");

    //thrust::device_ptr<int64_t> mask_dev_ptr = 
    //    thrust::device_pointer_cast(mask->data());
    //size_t out_len = thrust::count_if(mask_dev_ptr, 
    //    mask_dev_ptr + mask->numel(); ge_zero<T>());

    // FIXME: avoid mask copying 
    framework::LoDTensor mask_cpu;
    mask_cpu.Resize(mask->dims()); 
    mask_cpu.mutable_data<int64_t>(platform::CPUPlace());
    TensorCopySync(*mask, platform::CPUPlace(), &mask_cpu); 
   
    int64_t out_len = 0;
    for (size_t i = 0; i < mask->dims()[0]; ++i) {
      if (mask_cpu.data<int64_t>()[i] >= 0) {
        out_len += 1;
      }
    }
    
    ids->Resize({out_len, 1});
    offset->Resize({out_len, 1});
    ids->mutable_data<int64_t>(ctx.GetPlace());
    offset->mutable_data<int64_t>(ctx.GetPlace());

    auto x_dims = x->dims();
    auto out_dims = x_dims;
    out_dims[0] = out_len;
    out->Resize(out_dims);
    out->mutable_data<T>(ctx.GetPlace());

    framework::Tensor offset_cpu, ids_cpu;
    offset_cpu.Resize(offset->dims());
    offset_cpu.mutable_data<int64_t>(platform::CPUPlace());
    ids_cpu.Resize(ids->dims());
    ids_cpu.mutable_data<int64_t>(platform::CPUPlace());
    
    auto stream = ctx.cuda_device_context().stream();
    platform::CUDAPlace place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
     
    int64_t out_idx = 0;
    auto feat_num = x->numel() / x_dims[0];   
    for (int64_t i = 0; i < x_dims[0]; ++i) {
        if (mask_cpu.data<int64_t>()[i] >= 0) {
          offset_cpu.data<int64_t>()[out_idx] = i;
          ids_cpu.data<int64_t>()[out_idx] = mask_cpu.data<int64_t>()[i];
          memory::Copy(place, out->data<T>() + out_idx * feat_num, place,
                x->data<T>() + i * feat_num, feat_num * sizeof(T), stream);
          out_idx += 1;
        }
    }
    TensorCopySync(offset_cpu, ctx.GetPlace(), offset); 
    TensorCopySync(ids_cpu, ctx.GetPlace(), ids); 
    
  }
};


template <typename DeviceContext, typename T>
class MaskExtractGPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_out = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto* offset = ctx.Input<framework::LoDTensor>("Offset");
    auto* d_x = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));

    d_x->mutable_data<T>(ctx.GetPlace());
    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(ctx.template device_context<DeviceContext>(), d_x,
               static_cast<T>(0));
    auto x_dims = d_x->dims();
    auto feat_num = d_x->numel() / x_dims[0];   
    
    auto out_len = d_out->dims()[0];
    auto stream = ctx.cuda_device_context().stream();
    MaskExtractGradKernel<<<(out_len-1) / PADDLE_CUDA_NUM_THREADS + 1, 
        out_len, 0, stream>>>(
        d_out->data<T>(), offset->data<int64_t>(), feat_num, out_len, 
        d_x->data<T>());
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    mask_extract,
    ops::MaskExtractGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MaskExtractGPUKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MaskExtractGPUKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MaskExtractGPUKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    mask_extract_grad,
    ops::MaskExtractGPUGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MaskExtractGPUGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MaskExtractGPUGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MaskExtractGPUGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
