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

#include "paddle/fluid/operators/detection/yolo_box_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
static __global__ void GenDensityPriorBox(
    const int height, const int width, const int im_height, const int im_width,
    const T offset, const T step_width, const T step_height,
    const int num_priors, const T* ratios_shift, bool is_clip, const T var_xmin,
    const T var_ymin, const T var_xmax, const T var_ymax, T* out, T* var) {
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;
  int step_x = blockDim.x * gridDim.x;
  int step_y = blockDim.y * gridDim.y;
}

template <typename T>
class YoloBoxOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto* boxes = ctx.Output<Tensor>("Boxes");
    auto* scores = ctx.Output<Tensor>("Scores");

    auto anchors = ctx.Attr<std::vector<int>>("anchors");
    int class_num = ctx.Attr<int>("class_num");
    float conf_thresh = ctx.Attr<float>("conf_thresh");
    int downsample_ratio = ctx.Attr<int>("downsample_ratio");

    const int n = input->dims()[0];
    const int h = input->dims()[2];
    const int w = input->dims()[3];
    const int box_num = boxes->dims()[1];
    const int an_num = anchors.size() / 2;
    int input_size = downsample_ratio * h;

    const int stride = h * w;
    const int an_stride = (class_num + 5) * stride;

    const T* input_data = input->data<T>();
    T* boxes_data = boxes->mutable_data<T>({n}, ctx.GetPlace());
    memset(loss_data, 0, boxes->numel() * sizeof(T));
    T* scores_data = scores->mutable_data<T>({n}, ctx.GetPlace());
    memset(scores_data, 0, scores->numel() * sizeof(T));
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(density_prior_box,
                        ops::DensityPriorBoxOpCUDAKernel<float>,
                        ops::DensityPriorBoxOpCUDAKernel<double>);
