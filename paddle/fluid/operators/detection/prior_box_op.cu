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

#include "paddle/fluid/operators/detection/prior_box_op.h"

namespace paddle {
namespace operators {

template <typename T>
__device__ inline T clip(T in) {
  return min(max(in, 0.), 1.);
}

template <typename T>
__global__ void GenPriorBox(T* out, const T* aspect_ratios, const int height,
                            const int width, const int im_height,
                            const int im_width, const int as_num,
                            const T offset, const T step_width,
                            const T step_height, const T* min_sizes,
                            const T* max_sizes, const int min_num,
                            bool is_clip) {
  int num_priors = max_sizes ? as_num * min_num + min_num : as_num * min_num;
  int box_num = height * width * num_priors;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < box_num;
       i += blockDim.x * gridDim.x) {
    int h = i / (num_priors * width);
    int w = (i / num_priors) % width;
    int p = i % num_priors;
    int m = max_sizes ? p / (as_num + 1) : p / as_num;
    T cx = (w + offset) * step_width;
    T cy = (h + offset) * step_height;
    T bw, bh;
    T min_size = min_sizes[m];
    if (max_sizes) {
      int s = p % (as_num + 1);
      if (s < as_num) {
        T ar = aspect_ratios[s];
        bw = min_size * sqrt(ar) / 2.;
        bh = min_size / sqrt(ar) / 2.;
      } else {
        T max_size = max_sizes[m];
        bw = sqrt(min_size * max_size) / 2.;
        bh = bw;
      }
    } else {
      int s = p % as_num;
      T ar = aspect_ratios[s];
      bw = min_size * sqrt(ar) / 2.;
      bh = min_size / sqrt(ar) / 2.;
    }
    T xmin = (cx - bw) / im_width;
    T ymin = (cy - bh) / im_height;
    T xmax = (cx + bw) / im_width;
    T ymax = (cy + bh) / im_height;
    out[i * 4] = is_clip ? clip<T>(xmin) : xmin;
    out[i * 4 + 1] = is_clip ? clip<T>(ymin) : ymin;
    out[i * 4 + 2] = is_clip ? clip<T>(xmax) : xmax;
    out[i * 4 + 3] = is_clip ? clip<T>(ymax) : ymax;
  }
}

template <typename T>
__global__ void SetVariance(T* out, const T* var, const int vnum,
                            const int num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    out[i] = var[i % vnum];
  }
}

template <typename T>
class PriorBoxOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<paddle::framework::Tensor>("Input");
    auto* image = ctx.Input<paddle::framework::Tensor>("Image");
    auto* boxes = ctx.Output<paddle::framework::Tensor>("Boxes");
    auto* vars = ctx.Output<paddle::framework::Tensor>("Variances");

    auto min_sizes = ctx.Attr<std::vector<float>>("min_sizes");
    auto max_sizes = ctx.Attr<std::vector<float>>("max_sizes");
    auto input_aspect_ratio = ctx.Attr<std::vector<float>>("aspect_ratios");
    auto variances = ctx.Attr<std::vector<float>>("variances");
    auto flip = ctx.Attr<bool>("flip");
    auto clip = ctx.Attr<bool>("clip");

    std::vector<float> aspect_ratios;
    ExpandAspectRatios(input_aspect_ratio, flip, &aspect_ratios);

    T step_w = static_cast<T>(ctx.Attr<float>("step_w"));
    T step_h = static_cast<T>(ctx.Attr<float>("step_h"));
    T offset = static_cast<T>(ctx.Attr<float>("offset"));

    auto im_width = image->dims()[3];
    auto im_height = image->dims()[2];

    auto width = input->dims()[3];
    auto height = input->dims()[2];

    T step_width, step_height;
    if (step_w == 0 || step_h == 0) {
      step_width = static_cast<T>(im_width) / width;
      step_height = static_cast<T>(im_height) / height;
    } else {
      step_width = step_w;
      step_height = step_h;
    }

    int num_priors = aspect_ratios.size() * min_sizes.size();
    if (max_sizes.size() > 0) {
      num_priors += max_sizes.size();
    }
    int min_num = static_cast<int>(min_sizes.size());
    int box_num = width * height * num_priors;

    int block = 512;
    int grid = (box_num + block - 1) / block;

    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();

    boxes->mutable_data<T>(ctx.GetPlace());
    vars->mutable_data<T>(ctx.GetPlace());

    framework::Tensor r;
    framework::TensorFromVector(aspect_ratios, ctx.device_context(), &r);

    framework::Tensor min;
    framework::TensorFromVector(min_sizes, ctx.device_context(), &min);

    T* max_data = nullptr;
    framework::Tensor max;
    if (max_sizes.size() > 0) {
      framework::TensorFromVector(max_sizes, ctx.device_context(), &max);
      max_data = max.data<T>();
    }

    GenPriorBox<T><<<grid, block, 0, stream>>>(
        boxes->data<T>(), r.data<T>(), height, width, im_height, im_width,
        aspect_ratios.size(), offset, step_width, step_height, min.data<T>(),
        max_data, min_num, clip);

    framework::Tensor v;
    framework::TensorFromVector(variances, ctx.device_context(), &v);
    grid = (box_num * 4 + block - 1) / block;
    SetVariance<T><<<grid, block, 0, stream>>>(vars->data<T>(), v.data<T>(),
                                               variances.size(), box_num * 4);
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(prior_box, ops::PriorBoxOpCUDAKernel<float>,
                        ops::PriorBoxOpCUDAKernel<double>);
