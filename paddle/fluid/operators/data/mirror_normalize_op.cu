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

#include "paddle/fluid/operators/data/mirror_normalize_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace operators {
namespace data {

using framework::LoDTensor;

template<typename T>
__global__ void KeMirrorNormalize(
        const int numel, const T* in_data, const bool* mirrors, T* out_data,
        const float* mean, const float* std, const int chw, const int hw,
        const int w) {
  CUDA_KERNEL_LOOP(idx, numel) {
    int ni = idx / chw;
    int ci = (idx % chw) / hw;
    int wi = idx % w;
    
    int out_idx = idx;
    if (mirrors[ni]) out_idx = idx - 2 * wi + w - 1;
    out_data[out_idx] = (in_data[idx] - mean[ci]) / std[ci];
  }
}

template <typename T>
class MirrorNormalizeCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    LOG(ERROR) << "MirrorNormalizeCUDAKernel Compute start";
    auto* x = ctx.Input<LoDTensor>("X");
    auto* mirror  = ctx.Input<LoDTensor>("Mirror");
    auto* out = ctx.Output<LoDTensor>("Out");

    auto mean = ctx.Attr<std::vector<float>>("mean");
    auto std = ctx.Attr<std::vector<float>>("std");

    auto numel = x->numel();
    auto n = x->dims()[0];
    auto c = x->dims()[1];
    auto h = x->dims()[2];
    auto w = x->dims()[3];
    auto hw = h * w;
    auto chw = c * hw;

    const T* x_data = x->data<T>();
    const bool* mirror_data = mirror->data<bool>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.cuda_device_context();
    const auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
    const auto cplace = platform::CPUPlace();
    int bytes = sizeof(float) * mean.size();

    auto mean_ptr = memory::Alloc(dev_ctx, bytes);
    float* mean_data = reinterpret_cast<float*>(mean_ptr->ptr());
    memory::Copy(gplace, mean_data, cplace, mean.data(), bytes,
                 dev_ctx.stream());
    auto std_ptr = memory::Alloc(dev_ctx, bytes);
    float* std_data = reinterpret_cast<float*>(std_ptr->ptr());
    memory::Copy(gplace, std_data, cplace, std.data(), bytes,
                 dev_ctx.stream());

    platform::GpuLaunchConfig config =
                  platform::GetGpuLaunchConfig1D(dev_ctx, numel);
    KeMirrorNormalize<T><<<config.block_per_grid, config.thread_per_block,
                           0, dev_ctx.stream()>>>(
        numel, x_data, mirror_data, out_data, mean_data, std_data, chw, hw, w);
    LOG(ERROR) << "MirrorNormalizeCUDAKernel Compute finish";
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(mirror_normalize,
                        ops::data::MirrorNormalizeCUDAKernel<float>,
                        ops::data::MirrorNormalizeCUDAKernel<double>);
