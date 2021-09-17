/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_HIP
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif
#include <math.h>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/collective/c_clip_by_norm_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/gpu_launch_config.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename T>
__global__ void ClipByNormCUDACalcKernel(const T* src,
                                         const float* x_norm_square,
                                         const float max_norm, const int N,
                                         T* dst) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  float x_norm = sqrt(*x_norm_square);
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    const float src_tmp = static_cast<const float>(src[idx]);
    if (src_tmp <= max_norm) {
      dst[idx] = src[idx];
    } else {
      dst[idx] = static_cast<T>((src_tmp * max_norm) / x_norm);
    }
  }
}

template <typename Place, typename T>
class CollectiveClipByNormCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto max_norm = context.Attr<float>("max_norm");
    auto ring_id = context.Attr<int>("ring_id");
    auto in_tensors = context.MultiInput<Tensor>("X");
    auto out_tensors = context.MultiOutput<Tensor>("Out");
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    // var for allreduce
    const auto& place = context.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    gpuStream_t stream = nullptr;
    if (context.Attr<bool>("use_calc_stream")) {
      auto stream_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(stream_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    for (size_t i = 0; i < in_tensors.size(); ++i) {
      // calculate the squared_l2_norm value
      const Tensor* input = in_tensors[i];
      const int data_size = input->numel();

      std::vector<int> reduce_dims;
      reduce_dims.resize(input->dims().size());
      for (int i = 0; i < reduce_dims.size(); ++i) {
        reduce_dims[i] = i;
      }
      Tensor tmp =
          context.AllocateTmpTensor<float, platform::CUDADeviceContext>(
              {1}, dev_ctx);
      TensorReduceFunctorImpl<T, float, SquareSum>(*input, &tmp, reduce_dims,
                                                   dev_ctx.stream());
      float* tmp_data = tmp.data<float>();
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
          tmp_data, tmp_data, 1, platform::ToNCCLDataType(tmp.type()), ncclSum,
          comm->comm(), stream));
      if (!context.Attr<bool>("use_calc_stream")) {
#ifdef PADDLE_WITH_HIP
        PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamSynchronize(stream));
        PADDLE_ENFORCE_CUDA_SUCCESS(hipGetLastError());
#else
        PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
        PADDLE_ENFORCE_CUDA_SUCCESS(cudaGetLastError());
#endif
      }
      const T* input_data = input->data<T>();
      Tensor* output = out_tensors[i];
      T* output_data = output->mutable_data<T>(context.GetPlace());
      platform::GpuLaunchConfig config =
          platform::GetGpuLaunchConfig1D(dev_ctx, data_size);
      ClipByNormCUDACalcKernel<
          T><<<config.block_per_grid, config.thread_per_block, 0,
               dev_ctx.stream()>>>(input_data, tmp_data, max_norm, data_size,
                                   output_data);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    c_clip_by_norm,
    ops::CollectiveClipByNormCUDAKernel<plat::CUDADeviceContext, float>,
    ops::CollectiveClipByNormCUDAKernel<plat::CUDADeviceContext,
                                        plat::float16>);
