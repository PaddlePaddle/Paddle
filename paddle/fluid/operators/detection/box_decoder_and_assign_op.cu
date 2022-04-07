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

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/detection/box_decoder_and_assign_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void DecodeBoxKernel(const T* prior_box_data,
                                const T* prior_box_var_data,
                                const T* target_box_data, const int roi_num,
                                const int class_num, const T box_clip,
                                T* output_box_data) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < roi_num * class_num) {
    int i = idx / class_num;
    int j = idx % class_num;
    T prior_box_width = prior_box_data[i * 4 + 2] - prior_box_data[i * 4] + 1;
    T prior_box_height =
        prior_box_data[i * 4 + 3] - prior_box_data[i * 4 + 1] + 1;
    T prior_box_center_x = prior_box_data[i * 4] + prior_box_width / 2;
    T prior_box_center_y = prior_box_data[i * 4 + 1] + prior_box_height / 2;

    int offset = i * class_num * 4 + j * 4;
    T dw = prior_box_var_data[2] * target_box_data[offset + 2];
    T dh = prior_box_var_data[3] * target_box_data[offset + 3];
    if (dw > box_clip) {
      dw = box_clip;
    }
    if (dh > box_clip) {
      dh = box_clip;
    }
    T target_box_center_x = 0, target_box_center_y = 0;
    T target_box_width = 0, target_box_height = 0;
    target_box_center_x =
        prior_box_var_data[0] * target_box_data[offset] * prior_box_width +
        prior_box_center_x;
    target_box_center_y =
        prior_box_var_data[1] * target_box_data[offset + 1] * prior_box_height +
        prior_box_center_y;
    target_box_width = expf(dw) * prior_box_width;
    target_box_height = expf(dh) * prior_box_height;

    output_box_data[offset] = target_box_center_x - target_box_width / 2;
    output_box_data[offset + 1] = target_box_center_y - target_box_height / 2;
    output_box_data[offset + 2] =
        target_box_center_x + target_box_width / 2 - 1;
    output_box_data[offset + 3] =
        target_box_center_y + target_box_height / 2 - 1;
  }
}

template <typename T>
__global__ void AssignBoxKernel(const T* prior_box_data,
                                const T* box_score_data, T* output_box_data,
                                const int roi_num, const int class_num,
                                T* output_assign_box_data) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < roi_num) {
    int i = idx;
    T max_score = -1;
    int max_j = -1;
    for (int j = 0; j < class_num; ++j) {
      T score = box_score_data[i * class_num + j];
      if (score > max_score && j > 0) {
        max_score = score;
        max_j = j;
      }
    }
    if (max_j > 0) {
      for (int pno = 0; pno < 4; pno++) {
        output_assign_box_data[i * 4 + pno] =
            output_box_data[i * class_num * 4 + max_j * 4 + pno];
      }
    } else {
      for (int pno = 0; pno < 4; pno++) {
        output_assign_box_data[i * 4 + pno] = prior_box_data[i * 4 + pno];
      }
    }
  }
}

template <typename DeviceContext, typename T>
class BoxDecoderAndAssignCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* prior_box = context.Input<framework::LoDTensor>("PriorBox");
    auto* prior_box_var = context.Input<framework::Tensor>("PriorBoxVar");
    auto* target_box = context.Input<framework::LoDTensor>("TargetBox");
    auto* box_score = context.Input<framework::LoDTensor>("BoxScore");
    auto* output_box = context.Output<framework::Tensor>("DecodeBox");
    auto* output_assign_box =
        context.Output<framework::Tensor>("OutputAssignBox");

    auto roi_num = target_box->dims()[0];
    auto class_num = box_score->dims()[1];
    auto* target_box_data = target_box->data<T>();
    auto* prior_box_data = prior_box->data<T>();
    auto* prior_box_var_data = prior_box_var->data<T>();
    auto* box_score_data = box_score->data<T>();
    output_box->mutable_data<T>({roi_num, class_num * 4}, context.GetPlace());
    output_assign_box->mutable_data<T>({roi_num, 4}, context.GetPlace());
    T* output_box_data = output_box->data<T>();
    T* output_assign_box_data = output_assign_box->data<T>();

    int block = 512;
    int grid = (roi_num * class_num + block - 1) / block;
    auto& device_ctx = context.cuda_device_context();

    const T box_clip = context.Attr<T>("box_clip");

    DecodeBoxKernel<T><<<grid, block, 0, device_ctx.stream()>>>(
        prior_box_data, prior_box_var_data, target_box_data, roi_num, class_num,
        box_clip, output_box_data);

    context.device_context().Wait();
    int assign_grid = (roi_num + block - 1) / block;
    AssignBoxKernel<T><<<assign_grid, block, 0, device_ctx.stream()>>>(
        prior_box_data, box_score_data, output_box_data, roi_num, class_num,
        output_assign_box_data);
    context.device_context().Wait();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    box_decoder_and_assign,
    ops::BoxDecoderAndAssignCUDAKernel<paddle::platform::CUDADeviceContext,
                                       float>,
    ops::BoxDecoderAndAssignCUDAKernel<paddle::platform::CUDADeviceContext,
                                       double>);
