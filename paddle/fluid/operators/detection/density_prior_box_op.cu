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

#include "paddle/fluid/operators/detection/density_prior_box_op.h"

namespace paddle {
namespace operators {

template <typename T>
static __device__ inline T Clip(T in) {
  return min(max(in, 0.), 1.);
}

template <typename T>
static __global__ void GenDensityPriorBox(const int height,
                                          const int width,
                                          const int im_height,
                                          const int im_width,
                                          const T offset,
                                          const T step_width,
                                          const T step_height,
                                          const int num_priors,
                                          const T* ratios_shift,
                                          bool is_clip,
                                          const T var_xmin,
                                          const T var_ymin,
                                          const T var_xmax,
                                          const T var_ymax,
                                          T* out,
                                          T* var) {
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;
  int step_x = blockDim.x * gridDim.x;
  int step_y = blockDim.y * gridDim.y;

  const T* width_ratio = ratios_shift;
  const T* height_ratio = ratios_shift + num_priors;
  const T* width_shift = ratios_shift + 2 * num_priors;
  const T* height_shift = ratios_shift + 3 * num_priors;

  for (int j = gidy; j < height; j += step_y) {
    for (int i = gidx; i < width * num_priors; i += step_x) {
      int h = j;
      int w = i / num_priors;
      int k = i % num_priors;

      T center_x = (w + offset) * step_width;
      T center_y = (h + offset) * step_height;

      T center_x_temp = center_x + width_shift[k];
      T center_y_temp = center_y + height_shift[k];

      T box_width_ratio = width_ratio[k] / 2.;
      T box_height_ratio = height_ratio[k] / 2.;

      T xmin = max((center_x_temp - box_width_ratio) / im_width, 0.);
      T ymin = max((center_y_temp - box_height_ratio) / im_height, 0.);
      T xmax = min((center_x_temp + box_width_ratio) / im_width, 1.);
      T ymax = min((center_y_temp + box_height_ratio) / im_height, 1.);

      int out_offset = (j * width * num_priors + i) * 4;
      out[out_offset] = is_clip ? Clip<T>(xmin) : xmin;
      out[out_offset + 1] = is_clip ? Clip<T>(ymin) : ymin;
      out[out_offset + 2] = is_clip ? Clip<T>(xmax) : xmax;
      out[out_offset + 3] = is_clip ? Clip<T>(ymax) : ymax;

      var[out_offset] = var_xmin;
      var[out_offset + 1] = var_ymin;
      var[out_offset + 2] = var_xmax;
      var[out_offset + 3] = var_ymax;
    }
  }
}

template <typename T>
class DensityPriorBoxOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* image = ctx.Input<phi::DenseTensor>("Image");
    auto* boxes = ctx.Output<phi::DenseTensor>("Boxes");
    auto* vars = ctx.Output<phi::DenseTensor>("Variances");

    auto variances = ctx.Attr<std::vector<float>>("variances");
    auto is_clip = ctx.Attr<bool>("clip");

    auto fixed_sizes = ctx.Attr<std::vector<float>>("fixed_sizes");
    auto fixed_ratios = ctx.Attr<std::vector<float>>("fixed_ratios");
    auto densities = ctx.Attr<std::vector<int>>("densities");

    T step_w = static_cast<T>(ctx.Attr<float>("step_w"));
    T step_h = static_cast<T>(ctx.Attr<float>("step_h"));
    T offset = static_cast<T>(ctx.Attr<float>("offset"));

    auto img_width = image->dims()[3];
    auto img_height = image->dims()[2];

    auto feature_width = input->dims()[3];
    auto feature_height = input->dims()[2];

    T step_width, step_height;
    if (step_w == 0 || step_h == 0) {
      step_width = static_cast<T>(img_width) / feature_width;
      step_height = static_cast<T>(img_height) / feature_height;
    } else {
      step_width = step_w;
      step_height = step_h;
    }

    int num_priors = 0;
    for (size_t i = 0; i < densities.size(); ++i) {
      num_priors += (fixed_ratios.size()) * (pow(densities[i], 2));
    }
    int step_average = static_cast<int>((step_width + step_height) * 0.5);

    phi::DenseTensor h_temp;
    T* tdata = h_temp.mutable_data<T>({num_priors * 4}, platform::CPUPlace());
    int idx = 0;
    for (size_t s = 0; s < fixed_sizes.size(); ++s) {
      auto fixed_size = fixed_sizes[s];
      int density = densities[s];
      for (size_t r = 0; r < fixed_ratios.size(); ++r) {
        float ar = fixed_ratios[r];
        int shift = step_average / density;
        float box_width_ratio = fixed_size * sqrt(ar);
        float box_height_ratio = fixed_size / sqrt(ar);
        for (int di = 0; di < density; ++di) {
          for (int dj = 0; dj < density; ++dj) {
            float center_x_temp = shift / 2. + dj * shift - step_average / 2.;
            float center_y_temp = shift / 2. + di * shift - step_average / 2.;
            tdata[idx] = box_width_ratio;
            tdata[num_priors + idx] = box_height_ratio;
            tdata[2 * num_priors + idx] = center_x_temp;
            tdata[3 * num_priors + idx] = center_y_temp;
            idx++;
          }
        }
      }
    }

    boxes->mutable_data<T>(ctx.GetPlace());
    vars->mutable_data<T>(ctx.GetPlace());

    phi::DenseTensor d_temp;
    framework::TensorCopy(h_temp, ctx.GetPlace(), &d_temp);

    // At least use 32 threads, at most 512 threads.
    // blockx is multiple of 32.
    int blockx = std::min(
        static_cast<int64_t>(((feature_width * num_priors + 31) >> 5) << 5),
        static_cast<int64_t>(512L));
    int gridx = (feature_width * num_priors + blockx - 1) / blockx;
    dim3 threads(blockx, 1);
    dim3 grids(gridx, feature_height);

    auto stream = ctx.template device_context<phi::GPUContext>().stream();
    GenDensityPriorBox<T><<<grids, threads, 0, stream>>>(feature_height,
                                                         feature_width,
                                                         img_height,
                                                         img_width,
                                                         offset,
                                                         step_width,
                                                         step_height,
                                                         num_priors,
                                                         d_temp.data<T>(),
                                                         is_clip,
                                                         variances[0],
                                                         variances[1],
                                                         variances[2],
                                                         variances[3],
                                                         boxes->data<T>(),
                                                         vars->data<T>());
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(density_prior_box,
                        ops::DensityPriorBoxOpCUDAKernel<float>,
                        ops::DensityPriorBoxOpCUDAKernel<double>);
