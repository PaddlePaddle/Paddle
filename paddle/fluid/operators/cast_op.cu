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

#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename InT, typename OutT, int VecSize>
__global__ void VecCastCUDAKernel(const InT* in, const int64_t N, OutT* out) {
  using LoadT = platform::AlignedVector<InT, VecSize>;
  using StoreT = platform::AlignedVector<OutT, VecSize>;

  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int64_t i = idx * VecSize; i < N;
       i += blockDim.x * gridDim.x * VecSize) {
    LoadT in_val;
    platform::Load<InT, VecSize>(&in[i], &in_val);

    StoreT out_val;
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      out_val[j] = static_cast<OutT>(in_val[j]);
    }

    platform::Store<OutT, VecSize>(out_val, &out[i]);
  }
}

template <typename InT, typename OutT>
__global__ void CastCUDAKernel(const InT* in, const int64_t N, OutT* out) {
  CUDA_KERNEL_LOOP(index, N) { out[index] = static_cast<OutT>(in[index]); }
}

template <typename InT>
struct CastCUDAOpFunctor {
  const framework::Tensor* in_;
  framework::Tensor* out_;
  const platform::CUDADeviceContext& ctx_;
  CastCUDAOpFunctor(const framework::Tensor* in, framework::Tensor* out,
                    const platform::CUDADeviceContext& ctx)
      : in_(in), out_(out), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* in = in_->data<InT>();
    auto size = in_->numel();
    auto* out = out_->mutable_data<OutT>(ctx_.GetPlace());
    platform::GpuLaunchConfig config =
        platform::GetGpuLaunchConfig1D(ctx_, size);
    int vec_size = platform::GetVectorizedSize<OutT>(out);
    if (!std::is_same<InT, OutT>::value && vec_size == 4 && size % 4 == 0) {
      VecCastCUDAKernel<InT, OutT, 4><<<
          config.block_per_grid, config.thread_per_block, 0, ctx_.stream()>>>(
          in, size, out);
    } else {
      CastCUDAKernel<InT, OutT><<<config.block_per_grid,
                                  config.thread_per_block, 0, ctx_.stream()>>>(
          in, size, out);
    }
  }
};

template <typename InT>
class CastCUDAOpKernel : public framework::OpKernel<InT> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    framework::VisitDataType(
        static_cast<framework::proto::VarType::Type>(
            context.Attr<int>("out_dtype")),
        CastCUDAOpFunctor<InT>(
            in, out,
            context.template device_context<platform::CUDADeviceContext>()));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_CAST_CUDA_BASE(op_name, ...)                               \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      op_name, ops::CastCUDAOpKernel<float>, ops::CastCUDAOpKernel<double>, \
      ops::CastCUDAOpKernel<int>, ops::CastCUDAOpKernel<int64_t>,           \
      ops::CastCUDAOpKernel<int16_t>, ops::CastCUDAOpKernel<bool>,          \
      ops::CastCUDAOpKernel<uint8_t>, ops::CastCUDAOpKernel<plat::float16>, \
      ops::CastCUDAOpKernel<plat::complex<float>>,                          \
      ops::CastCUDAOpKernel<plat::complex<double>>, ##__VA_ARGS__);

#if !defined(PADDLE_WITH_HIP)
REGISTER_CAST_CUDA_BASE(cast, ops::CastCUDAOpKernel<plat::bfloat16>)
// See [ why register transfer_dtype_op alias with cast_op? ] in cast_op.cc
REGISTER_CAST_CUDA_BASE(transfer_dtype, ops::CastCUDAOpKernel<plat::bfloat16>)
#else
REGISTER_CAST_CUDA_BASE(cast)
REGISTER_CAST_CUDA_BASE(transfer_dtype)
#endif
