/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/box_clip_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTenso = framework::LoDTensor;

static constexpr int ImInfoSize = 3;

template <typename T, int BlockSize>
static __global__ void GPUBoxClip(const T *input, const size_t *lod,
                                  const size_t width, const T *im_info,
                                  T *output) {
  T im_w = round(im_info[blockIdx.x * ImInfoSize + 1] /
                 im_info[blockIdx.x * ImInfoSize + 2]);
  T im_h = round(im_info[blockIdx.x * ImInfoSize] /
                 im_info[blockIdx.x * ImInfoSize + 2]);
  for (int i = threadIdx.x; i < (lod[blockIdx.x + 1] - lod[blockIdx.x]) * width;
       i += BlockSize) {
    int idx = lod[blockIdx.x] * width + i;
    T im_size = (idx % 2 == 0) ? im_w : im_h;
    output[idx] = max(min(input[idx], im_size - 1), T(0.));
  }
}

template <typename DeviceContext, typename T>
class GPUBoxClipKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *input = context.Input<LoDTensor>("Input");
    auto *im_info = context.Input<Tensor>("ImInfo");
    auto *output = context.Output<LoDTensor>("Output");
    const int64_t num = input->dims()[0];
    const int64_t bbox_width = input->numel() / num;
    auto lod = input->lod();
    framework::LoD abs_offset_lod = framework::ToAbsOffset(lod);
    auto &dev_ctx = context.template device_context<DeviceContext>();
    auto stream = dev_ctx.stream();
    const size_t batch_size = lod.back().size() - 1;
    T *output_data = output->mutable_data<T>(dev_ctx.GetPlace());
    paddle::framework::MixVector<size_t> mix_vector(&abs_offset_lod[0]);
    GPUBoxClip<T, 512><<<batch_size, 512, 0, stream>>>(
        input->data<T>(), mix_vector.CUDAMutableData(dev_ctx.GetPlace()),
        bbox_width, im_info->data<T>(), output_data);
    mix_vector.CopyToCPU();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    box_clip, ops::GPUBoxClipKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUBoxClipKernel<paddle::platform::CUDADeviceContext, double>);
