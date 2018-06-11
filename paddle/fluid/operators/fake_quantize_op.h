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
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FakeQuantizeKernel : public framework::OpKernel<T> {
 public:
  T find_abs_max(framework::Tensor* in, int n) const {
    T* p = in->mutable_data<T>(platform::CPUPlace());
    T abs_max = (T)0.00000001;
    for (int i = 0; i < n; i++) {
      T tmp = fabs(p[i]);
      if (tmp > abs_max) abs_max = tmp;
    }
    return T(abs_max);
  }

  T find_range_abs_max(framework::Tensor* scale_list,
                       framework::Tensor* out_scale, const T& cur_scale,
                       int window_size, int current_iter) const {
    T* sl = scale_list->mutable_data<T>(platform::CPUPlace());
    T remove_tmp = sl[current_iter];
    sl[current_iter] = cur_scale;
    T& max_scale = out_scale->mutable_data<T>(platform::CPUPlace())[0];
    if (max_scale < cur_scale) {
      max_scale = cur_scale;
    } else if (fabs(remove_tmp - max_scale) < 1e-6) {
      int size = (current_iter > window_size) ? window_size : current_iter;
      max_scale = T(find_abs_max(scale_list, size));
    }
    return max_scale;
  }

  T find_moving_average_abs_max(framework::Tensor* in_scale,
                                framework::Tensor* out_scale,
                                const T& cur_scale) const {
    T* ins = in_scale->mutable_data<T>(platform::CPUPlace());
    T* outs = out_scale->mutable_data<T>(platform::CPUPlace());
    outs[0] = 0.9 * cur_scale + 0.1 * ins[0];
    return T(outs[0]);
  }

  int apply_saturate(framework::Tensor* src, framework::Tensor* dst,
                     const T& min, const T& max) const {
    T* in = src->mutable_data<T>(platform::CPUPlace());
    T* out = dst->mutable_data<T>(platform::CPUPlace());
    int num_saturate = 0;
    int n = src->numel();
    for (int i = 0; i < n; ++i) {
      if (in[i] > max) {
        out[i] = max;
        ++num_saturate;
      } else if (in[i] < min) {
        out[i] = min;
        ++num_saturate;
      } else {
        out[i] = in[i];
      }
    }
    return num_saturate;
  }

  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* tensor = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    const bool is_test = context.Attr<bool>("is_test");
    tensor->mutable_data<T>(in->place());

    auto* oms_tensor = context.Output<framework::Tensor>("OutMovingScale");
    oms_tensor->mutable_data<T>(in->place());

    auto quantize_type =
        static_cast<std::string>(context.Attr<std::string>("quantize_type"));
    if (quantize_type == std::string("range_abs_max")) {
      auto* oss_tensor = context.Output<framework::Tensor>("OutScales");
      oss_tensor->mutable_data<T>(
          context.Input<framework::Tensor>("InScales")->place());
      auto* oci_tensor = context.Output<framework::Tensor>("OutCurrentIter");
      oci_tensor->mutable_data<T>(
          context.Input<framework::Tensor>("InCurrentIter")->place());
    }

    T scale = T(1);
    int window_size = context.Attr<int>("window_size");
    int bit_length = context.Attr<int>("bit_length");
    int bin_cnt = std::pow(2, bit_length - 1) - 1;
    LOG(ERROR) << "bin_cnt:" << bin_cnt;

    if (quantize_type == std::string("abs_max")) {
      auto* saving_scale = context.Output<framework::Tensor>("OutMovingScale");
      scale = T(find_abs_max(const_cast<framework::Tensor*>(in), in->numel()));
      saving_scale->mutable_data<T>(platform::CPUPlace())[0] = scale;

      auto& device_ctx = context.template device_context<DeviceContext>();
      auto* scale_list = context.Output<framework::Tensor>("OutScales");
      math::SetConstant<DeviceContext, T> scalar;
      scale_list->mutable_data<T>(context.GetPlace());
      scalar(device_ctx, scale_list, static_cast<T>(0));
      auto* iter = context.Output<framework::Tensor>("OutCurrentIter");
      iter->mutable_data<T>(context.GetPlace());
      scalar(device_ctx, iter, static_cast<T>(0));
    } else if (quantize_type == std::string("range_abs_max")) {
      auto* moving_scale = context.Input<framework::Tensor>("InMovingScale");
      if (is_test) {
        scale = moving_scale->data<T>()[0];
      } else {
        auto* it = context.Input<framework::Tensor>("InCurrentIter");
        auto* iter = context.Output<framework::Tensor>("OutCurrentIter");
        const int* last_iter = it->data<int>();
        int* current_iter = iter->mutable_data<int>(platform::CPUPlace());
        auto* scale_list = context.Output<framework::Tensor>("OutScales");
        auto* saving_scale =
            context.Output<framework::Tensor>("OutMovingScale");
        scale = find_abs_max(const_cast<framework::Tensor*>(in), in->numel());
        scale = find_range_abs_max(scale_list, saving_scale, scale, window_size,
                                   current_iter[0]);
        (*current_iter) = (*last_iter) + 1;
      }
    } else if (quantize_type == std::string("range_abs_max")) {
      auto* moving_scale = context.Input<framework::Tensor>("InMovingScale");
      if (is_test) {
        scale = moving_scale->data<T>()[0];
      } else {
        scale = find_abs_max(const_cast<framework::Tensor*>(in), in->numel());
        auto* saving_scale =
            context.Output<framework::Tensor>("OutMovingScale");
        scale = find_moving_average_abs_max(
            const_cast<framework::Tensor*>(moving_scale), saving_scale, scale);
      }
    }

    apply_saturate(const_cast<framework::Tensor*>(in), tensor, -scale, scale);

    auto& dev =
        *context.template device_context<DeviceContext>().eigen_device();
    auto eigen_out = framework::EigenVector<T>::Flatten(*tensor);
    auto eigen_in = framework::EigenVector<T>::Flatten(*tensor);
    eigen_out.device(dev) = (bin_cnt / scale * eigen_in).round();
  }
};

}  // namespace operators
}  // namespace paddle
