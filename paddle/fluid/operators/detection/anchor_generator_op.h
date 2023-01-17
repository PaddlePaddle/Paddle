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

#pragma once
#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_CUDA
template <typename T>
extern __global__ void GenAnchors(T* out,
                                  const T* aspect_ratios,
                                  const int ar_num,
                                  const T* anchor_sizes,
                                  const int as_num,
                                  const T* stride,
                                  const int sd_num,
                                  const int height,
                                  const int width,
                                  const T offset);

template <typename T>
extern __global__ void SetVariance(T* out,
                                   const T* var,
                                   const int vnum,
                                   const int num);
#endif

template <typename T>
class AnchorGeneratorOpKernel : public framework::OpKernel<T> {
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

    auto feature_width = input->dims()[3];
    auto feature_height = input->dims()[2];

    T stride_width, stride_height;
    stride_width = stride[0];
    stride_height = stride[1];

    int num_anchors = aspect_ratios.size() * anchor_sizes.size();

    anchors->mutable_data<T>(ctx.GetPlace());
    vars->mutable_data<T>(ctx.GetPlace());

    auto e_anchors = phi::EigenTensor<T, 4>::From(*anchors);
    for (int h_idx = 0; h_idx < feature_height; ++h_idx) {
      for (int w_idx = 0; w_idx < feature_width; ++w_idx) {
        T x_ctr = (w_idx * stride_width) + offset * (stride_width - 1);
        T y_ctr = (h_idx * stride_height) + offset * (stride_height - 1);
        T area, area_ratios;
        T base_w, base_h;
        T scale_w, scale_h;
        T anchor_width, anchor_height;
        int idx = 0;
        for (size_t r = 0; r < aspect_ratios.size(); ++r) {
          auto ar = aspect_ratios[r];
          for (size_t s = 0; s < anchor_sizes.size(); ++s) {
            auto anchor_size = anchor_sizes[s];
            area = stride_width * stride_height;
            area_ratios = area / ar;
            base_w = round(sqrt(area_ratios));
            base_h = round(base_w * ar);
            scale_w = anchor_size / stride_width;
            scale_h = anchor_size / stride_height;
            anchor_width = scale_w * base_w;
            anchor_height = scale_h * base_h;
            e_anchors(h_idx, w_idx, idx, 0) =
                (x_ctr - 0.5 * (anchor_width - 1));
            e_anchors(h_idx, w_idx, idx, 1) =
                (y_ctr - 0.5 * (anchor_height - 1));
            e_anchors(h_idx, w_idx, idx, 2) =
                (x_ctr + 0.5 * (anchor_width - 1));
            e_anchors(h_idx, w_idx, idx, 3) =
                (y_ctr + 0.5 * (anchor_height - 1));
            idx++;
          }
        }
      }
    }

    phi::DenseTensor var_t;
    var_t.mutable_data<T>(
        phi::make_ddim({1, static_cast<int>(variances.size())}),
        ctx.GetPlace());
    auto var_et = phi::EigenTensor<T, 2>::From(var_t);
    for (size_t i = 0; i < variances.size(); ++i) {
      var_et(0, i) = variances[i];
    }

    int anchor_num = feature_height * feature_width * num_anchors;
    auto var_dim = vars->dims();
    vars->Resize({anchor_num, static_cast<int>(variances.size())});

    auto e_vars = phi::EigenMatrix<T, Eigen::RowMajor>::From(*vars);
    e_vars = var_et.broadcast(Eigen::DSizes<int, 2>(anchor_num, 1));

    vars->Resize(var_dim);
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle
