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
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTenso = framework::LoDTensor;

static constexpr int ImShapeSize = 2;

template <typename T, int BlockSize>
static __global__ void GPUBoxClip(const T *input, const size_t *lod,
                                  const size_t width, const T *im_shape,
                                  T *output) {
  T im_w = im_shape[blockIdx.x * ImShapeSize + 1];
  T im_h = im_shape[blockIdx.x * ImShapeSize];
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
    auto *im_shape = context.Input<Tensor>("ImShape");
    auto *output = context.Output<LoDTensor>("Output");
    const int64_t num = input->dims()[0];
    const int64_t bbox_width = input->numel() / num;

    auto &dev_ctx = context.template device_context<DeviceContext>();
    auto stream = dev_ctx.stream();
    // LOG(ERROR)<<"DEBUG111111111111";
    T *output_data = output->mutable_data<T>(dev_ctx.GetPlace());
    if (context.HasInput("RoisNum")) {
      auto *rois_num = context.Input<Tensor>("RoisNum");
      // LOG(ERROR)<<"DEBUG rois_num"<<rois_num;
      const size_t batch_size = rois_num->numel();
      const size_t *rois_num_data =
          reinterpret_cast<const size_t *>(rois_num->data<int>());
      GPUBoxClip<T, 512><<<batch_size, 512, 0, stream>>>(
          input->data<T>(), rois_num_data, bbox_width, im_shape->data<T>(),
          output_data);
      // LOG(ERROR)<<"DEBUG33333333333";
      // LOG(ERROR)<<"DEBUG output_data:"<<output_data;
    } else {
      auto lod = input->lod();
      framework::LoD abs_offset_lod = framework::ToAbsOffset(lod);
      const size_t batch_size = lod.back().size() - 1;
      GPUBoxClip<T, 512><<<batch_size, 512, 0, stream>>>(
          input->data<T>(),
          abs_offset_lod[0].CUDAMutableData(dev_ctx.GetPlace()), bbox_width,
          im_shape->data<T>(), output_data);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    box_clip, ops::GPUBoxClipKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUBoxClipKernel<paddle::platform::CUDADeviceContext, double>);
