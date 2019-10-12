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
#include "paddle/fluid/platform/transform.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using platform::Transform;

template <typename DeviceContext, typename T>
class PReluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* alpha = context.Input<Tensor>("Alpha");
    auto* out = context.Output<Tensor>("Out");

    const T* x_ptr = x->data<T>();
    T* o_ptr = out->mutable_data<T>(context.GetPlace());

    const T* alpha_ptr = alpha->data<T>();
    auto& mode = context.Attr<std::string>("mode");

    int numel = x->numel();
    auto dim = x->dims();
    int index = 0;
    int i = 0;
    if (mode == "channel") {
      int temp = numel / (dim[0] * dim[1]);
      for (i = 0; i < numel; i++) {
        index = (i / temp) % dim[1];
        o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[index] * x_ptr[i];
      }
    } else if (mode == "element") {
      for (i = 0; i < numel; i++) {
        o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[i] * x_ptr[i];
      }
    } else {
      for (i = 0; i < numel; i++) {
        o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[0] * x_ptr[i];
      }
    }
  }
};

template <typename DeviceContext, typename T>
class PReluGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dalpha = context.Output<Tensor>(framework::GradVarName("Alpha"));
    auto* out = context.Input<Tensor>("Out");
    auto* alpha = context.Input<Tensor>("Alpha");
    const T* alpha_ptr = alpha->data<T>();
    const T* x_ptr = x->data<T>();
    const T* dout_ptr = dout->data<T>();
    const T* out_ptr = out->data<T>();
    std::string mode = context.Attr<std::string>("mode");
    int numel = x->numel();
    auto dim = x->dims();
    int index = 0;
    int i = 0;
    int temp = 0;
    if (dx) {
      T* dx_ptr = dx->mutable_data<T>(context.GetPlace());
      if (mode == "channel") {
        for (i = 0; i < numel; i++) {
          temp = numel / (dim[0] * dim[1]);
          index = (i / temp) % dim[1];
          dx_ptr[i] =
              out_ptr[i] > 0 ? dout_ptr[i] : alpha_ptr[index] * dout_ptr[i];
        }
      } else if (mode == "element") {
        for (i = 0; i < numel; i++) {
          dx_ptr[i] = out_ptr[i] > 0 ? dout_ptr[i] : alpha_ptr[i] * dout_ptr[i];
        }
      } else {
        for (i = 0; i < numel; i++) {
          dx_ptr[i] = out_ptr[i] > 0 ? dout_ptr[i] : alpha_ptr[0] * dout_ptr[i];
        }
      }
    }

    index = 0;
    if (dalpha) {
      T* dalpha_ptr = dalpha->mutable_data<T>(context.GetPlace());
      memset(dalpha_ptr, 0, sizeof(T) * dalpha->numel());

      if (mode == "channel") {
        for (i = 0; i < numel; i++) {
          temp = numel / (dim[0] * dim[1]);
          index = (i / temp) % dim[1];
          dalpha_ptr[index] += out_ptr[i] > 0 ? 0 : x_ptr[i] * dout_ptr[i];
        }
      } else if (mode == "element") {
        for (i = 0; i < numel; i++) {
          dalpha_ptr[i] += out_ptr[i] > 0 ? 0 : x_ptr[i] * dout_ptr[i];
        }
      } else {
        for (i = 0; i < numel; i++) {
          dalpha_ptr[0] += out_ptr[i] > 0 ? 0 : x_ptr[i] * dout_ptr[i];
        }
      }
    }

    // TODO(Guanzhong): add GPU kernels
  }
};

}  // namespace operators
}  // namespace paddle
