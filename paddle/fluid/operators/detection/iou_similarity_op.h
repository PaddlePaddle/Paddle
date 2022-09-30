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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

template <typename T>
inline HOSTDEVICE T IOUSimilarity(T xmin1,
                                  T ymin1,
                                  T xmax1,
                                  T ymax1,
                                  T xmin2,
                                  T ymin2,
                                  T xmax2,
                                  T ymax2,
                                  bool normalized,
                                  T eps) {
  constexpr T zero = static_cast<T>(0);
  T area1;
  T area2;
  if (!normalized) {
    area1 = (ymax1 - ymin1 + 1) * (xmax1 - xmin1 + 1);
    area2 = (ymax2 - ymin2 + 1) * (xmax2 - xmin2 + 1);
  } else {
    area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
    area2 = (ymax2 - ymin2) * (xmax2 - xmin2);
  }

  T inter_xmax = xmax1 > xmax2 ? xmax2 : xmax1;
  T inter_ymax = ymax1 > ymax2 ? ymax2 : ymax1;
  T inter_xmin = xmin1 > xmin2 ? xmin1 : xmin2;
  T inter_ymin = ymin1 > ymin2 ? ymin1 : ymin2;
  T inter_height = inter_ymax - inter_ymin;
  T inter_width = inter_xmax - inter_xmin;
  if (!normalized) {
    inter_height = inter_height + 1;
    inter_width = inter_width + 1;
  }
  inter_height = inter_height > zero ? inter_height : zero;
  inter_width = inter_width > zero ? inter_width : zero;
  T inter_area = inter_width * inter_height;
  T union_area = area1 + area2 - inter_area + eps;
  T sim_score = inter_area / union_area;
  return sim_score;
}

template <typename T>
struct IOUSimilarityFunctor {
  IOUSimilarityFunctor(
      const T* x, const T* y, T* z, int cols, bool normalized, T eps)
      : x_(x),
        y_(y),
        z_(z),
        cols_(static_cast<size_t>(cols)),
        normalized_(normalized),
        eps_(eps) {}

  inline HOSTDEVICE void operator()(size_t tid) const {
    size_t row_id = tid / cols_;
    size_t col_id = tid % cols_;

    T x_min1 = x_[row_id * 4];
    T y_min1 = x_[row_id * 4 + 1];
    T x_max1 = x_[row_id * 4 + 2];
    T y_max1 = x_[row_id * 4 + 3];

    T x_min2 = y_[col_id * 4];
    T y_min2 = y_[col_id * 4 + 1];
    T x_max2 = y_[col_id * 4 + 2];
    T y_max2 = y_[col_id * 4 + 3];

    T sim = IOUSimilarity(x_min1,
                          y_min1,
                          x_max1,
                          y_max1,
                          x_min2,
                          y_min2,
                          x_max2,
                          y_max2,
                          normalized_,
                          eps_);

    z_[row_id * cols_ + col_id] = sim;
  }
  const T* x_;
  const T* y_;
  T* z_;
  const size_t cols_;
  bool normalized_;
  T eps_;
};

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class IOUSimilarityKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::LoDTensor* in_x = ctx.Input<framework::LoDTensor>("X");
    const phi::DenseTensor* in_y = ctx.Input<phi::DenseTensor>("Y");
    bool normalized = ctx.Attr<bool>("box_normalized");
    framework::LoDTensor* out = ctx.Output<framework::LoDTensor>("Out");

    int x_n = in_x->dims()[0];
    int y_n = in_y->dims()[0];
    T eps = static_cast<T>(1e-10);
    IOUSimilarityFunctor<T> functor(in_x->data<T>(),
                                    in_y->data<T>(),
                                    out->mutable_data<T>(ctx.GetPlace()),
                                    y_n,
                                    normalized,
                                    eps);

    platform::ForRange<DeviceContext> for_range(
        static_cast<const DeviceContext&>(ctx.device_context()), x_n * y_n);
    for_range(functor);
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle
