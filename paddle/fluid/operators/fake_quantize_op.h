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

#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/clip_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
struct FindAbsMaxFunctor {
  void operator()(const DeviceContext& ctx, const T* in, const int num, T* out);
};

template <typename DeviceContext, typename T>
struct ClipAndFakeQuantFunctor {
  void operator()(const DeviceContext& ctx, const framework::Tensor& in,
                  const framework::Tensor* scale, const int bin_cnt,
                  framework::Tensor* out);
};

template <typename DeviceContext, typename T>
struct FindRangeAbsMaxFunctor {
  void operator()(const DeviceContext& ctx, const framework::Tensor& in,
                  const framework::Tensor& cur_scale,
                  const framework::Tensor& last_scale,
                  const framework::Tensor& iter, const int window_size,
                  framework::Tensor* scales_arr, framework::Tensor* out_scale,
                  framework::Tensor* out);
};

void FindRangeAbsMax(const platform::CUDADeviceContext& ctx,
                     framework::Tensor* scale_list, const T last_max_scale,
                     const T& cur_scale, int window_size,
                     int current_iter) const {
  T* sl = scale_list->mutable_data<T>(scale_list->place());
  T remove_tmp;
  auto& gpu_place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
  int idx = current_iter % window_size;
  memory::Copy(platform::CPUPlace(), &remove_tmp, gpu_place, sl + idx,
               sizeof(float), ctx.stream());
  memory::Copy(gpu_place, sl + idx, platform::CPUPlace(), &cur_scale, sizeof(T),
               ctx.stream());
  T max_scale = last_max_scale;
  if (max_scale < cur_scale) {
    max_scale = cur_scale;
  } else if (fabs(remove_tmp - max_scale) < 1e-6) {
    int size = (current_iter > window_size) ? window_size : current_iter;
    max_scale = T(FindAbsMaxGpu(ctx, scale_list->data<float>(), size));
  }
  return max_scale;
}

template <typename DeviceContext, typename T>
class FakeQuantizeAbsMaxKernel : public framework::OpKernel<T> {
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* in = context.Input<framework::Tensor>("X");
    auto* in_scale = context.Input<framework::Tensor>("InScale");

    auto* out = context.Output<framework::Tensor>("Out");
    auto* out_scale = context.Output<framework::Tensor>("OutScale");
    T* out_data = out->mutable_data<T>(context.GetPlace());
    T* out_s = out_scale->mutable_data<T>(context.GetPlace());

    int bit_length = context.Attr<int>("bit_length");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;

    auto& dev_ctx = context.template device_context<DeviceContext>();
    const T* in_data = in->data<T>();
    FindAbsMaxFunctor<DeviceContext, T>()(dev_ctx, in_data, in.numel(), out_s);
    ClipAndFakeQuantFunctor<DeviceContext, T>()(dev_ctx, *in, *out_scale,
                                                bin_cnt, out);
  }
};

template <typename DeviceContext, typename T>
class FakeQuantizeRangeAbsMaxKernel : public framework::OpKernel<T> {
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* in = context.Input<framework::Tensor>("X");
    auto* in_scale = context.Input<framework::Tensor>("X");

    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    bool is_test = context.Attr<bool>("is_test");
    int bit_length = context.Attr<int>("bit_length");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;
    auto& dev_ctx = context.template device_context<DeviceContext>();

    // testing
    if (is_test) {
      ClipAndFakeQuantFunctor<DeviceContext, T>()(dev_ctx, *in, *in_scale,
                                                  bin_cnt, out);
      return;
    }

    // training
    auto* out_scale = context.Output<framework::Tensor>("OutScale");
    auto* in_scales = context.Input<framework::Tensor>("InScales");
    auto* out_scales = context.Input<framework::Tensor>("OutScales");
    auto* iter = context.Input<framework::Tensor>("Iter");

    bool window_size = context.Attr<bool>("window_size");
    out_scale->mutable_data<T>(context.GetPlace());

    Tensor cur_scale;
    T* cur_scale_data = cur_scale.mutable_data<T>({1}, context.GetPlace());
    FindAbsMaxFunctor<DeviceContext, T>()(dev_ctx, in->data<T>(), in->numel(),
                                          cur_scale_data);
    FindRangeAbsMaxFunctor<DeviceContext, T>()(
        dev_ctx, cur_scale, in_scale, iter, window_size, out_scale, out_scale);
    ClipAndFakeQuantFunctor<DeviceContext, T>()(dev_ctx, *in, *out_scale,
                                                bin_cnt, out);
  }
};

}  // namespace operators
}  // namespace paddle
