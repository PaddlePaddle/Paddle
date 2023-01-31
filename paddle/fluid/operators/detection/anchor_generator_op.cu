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

#include "paddle/fluid/operators/detection/anchor_generator_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void GenAnchors(T* out,
                           const T* aspect_ratios,
                           const int ar_num,
                           const T* anchor_sizes,
                           const int as_num,
                           const T* stride,
                           const int sd_num,
                           const int height,
                           const int width,
                           const T offset) {
  int num_anchors = as_num * ar_num;
  int box_num = height * width * num_anchors;
  CUDA_KERNEL_LOOP(i, box_num) {
    int h_idx = i / (num_anchors * width);
    int w_idx = (i / num_anchors) % width;
    T stride_width = stride[0];
    T stride_height = stride[1];
    T x_ctr = (w_idx * stride_width) + offset * (stride_width - 1);
    T y_ctr = (h_idx * stride_height) + offset * (stride_height - 1);
    T area, area_ratios;
    T base_w, base_h;
    T scale_w, scale_h;
    T anchor_width, anchor_height;
    int anch_idx = i % num_anchors;
    int ar_idx = anch_idx / as_num;
    int as_idx = anch_idx % as_num;
    T aspect_ratio = aspect_ratios[ar_idx];
    T anchor_size = anchor_sizes[as_idx];
    area = stride_width * stride_height;
    area_ratios = area / aspect_ratio;
    base_w = round(sqrt(area_ratios));
    base_h = round(base_w * aspect_ratio);
    scale_w = anchor_size / stride_width;
    scale_h = anchor_size / stride_height;
    anchor_width = scale_w * base_w;
    anchor_height = scale_h * base_h;

    T xmin = (x_ctr - .5f * (anchor_width - 1));
    T ymin = (y_ctr - .5f * (anchor_height - 1));
    T xmax = (x_ctr + .5f * (anchor_width - 1));
    T ymax = (y_ctr + .5f * (anchor_height - 1));
    reinterpret_cast<float4*>(out)[i] = make_float4(xmin, ymin, xmax, ymax);
  }
}

template <typename T>
__global__ void SetVariance(T* out,
                            const T* var,
                            const int vnum,
                            const int num) {
  CUDA_KERNEL_LOOP(i, num) { out[i] = var[i % vnum]; }
}

template <typename T>
class AnchorGeneratorOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* anchors = ctx.Output<phi::DenseTensor>("Anchors");
    auto* vars = ctx.Output<phi::DenseTensor>("Variances");

    auto anchor_sizes = ctx.Attr<std::vector<float>>("anchor_sizes");
    auto aspect_ratios = ctx.Attr<std::vector<float>>("aspect_ratios");
    auto stride = ctx.Attr<std::vector<float>>("stride");
    auto variances = ctx.Attr<std::vector<float>>("variances");

    T offset = static_cast<T>(ctx.Attr<float>("offset"));

    auto width = input->dims()[3];
    auto height = input->dims()[2];

    int num_anchors = aspect_ratios.size() * anchor_sizes.size();

    int box_num = width * height * num_anchors;

    int block = 512;
    int grid = (box_num + block - 1) / block;

    auto stream = ctx.template device_context<phi::GPUContext>().stream();

    anchors->mutable_data<T>(ctx.GetPlace());
    vars->mutable_data<T>(ctx.GetPlace());

    phi::DenseTensor ar;
    framework::TensorFromVector(aspect_ratios, ctx.device_context(), &ar);

    phi::DenseTensor as;
    framework::TensorFromVector(anchor_sizes, ctx.device_context(), &as);

    phi::DenseTensor sd;
    framework::TensorFromVector(stride, ctx.device_context(), &sd);

    GenAnchors<T><<<grid, block, 0, stream>>>(anchors->data<T>(),
                                              ar.data<T>(),
                                              aspect_ratios.size(),
                                              as.data<T>(),
                                              anchor_sizes.size(),
                                              sd.data<T>(),
                                              stride.size(),
                                              height,
                                              width,
                                              offset);

    phi::DenseTensor v;
    framework::TensorFromVector(variances, ctx.device_context(), &v);
    grid = (box_num * 4 + block - 1) / block;
    SetVariance<T><<<grid, block, 0, stream>>>(
        vars->data<T>(), v.data<T>(), variances.size(), box_num * 4);
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(anchor_generator,
                        ops::AnchorGeneratorOpCUDAKernel<float>,
                        ops::AnchorGeneratorOpCUDAKernel<double>);
