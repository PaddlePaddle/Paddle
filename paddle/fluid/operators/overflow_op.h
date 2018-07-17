// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

// isinf isnan in std has been customized.
template <typename T>
struct InfinityFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE bool operator()(const T& a) { return std::isinf(a); }
};

template <typename T>
struct NANFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE bool operator()(const T& a) { return std::isnan(a); }
};

template <typename T>
struct OverflowFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE bool operator()(const T& a) {
    return std::isnan(a) || std::isinf(a);
  }
};

template <typename T>
struct NotOverflowFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE bool operator()(const T& a) {
    return !std::isnan(a) && !std::isinf(a);
  }
};

template <typename DeviceContext, typename T, typename Functor>
class OverflowKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* x = ctx.InputVar("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    platform::Transform<DeviceContext> transfunctor;
    auto* out_begin = out->mutable_data<T>(ctx.GetPlace());
    if (x->IsType<framework::LoDTensor>()) {
      auto* in = ctx.Input<framework::Tensor>("X");
      auto* in_begin = in->data<T>();
      auto numel = in->numel();
      auto* in_end = in_begin + numel;
      transfunctor(ctx.template device_context<DeviceContext>(), in_begin,
                   in_end, out_begin, Functor());
    } else if (x->IsType<framework::SelectedRows>()) {
      auto& in = ctx.Input<framework::SelectedRows>("X")->value();
      auto* in_begin = in.data<T>();
      auto numel = in.numel();
      auto* in_end = in_begin + numel;
      transfunctor(ctx.template device_context<DeviceContext>(), in_begin,
                   in_end, out_begin, Functor());
    } else {
      PADDLE_THROW("Unsupported input type.");
    }
  }
};

}  // namespace operators
}  // namespace paddle

#define FOR_EACH_KERNEL_FUNCTOR(__macro) \
  __macro(isinf, InfinityFunctor);       \
  __macro(isnan, NANFunctor);            \
  __macro(overflow, OverflowFunctor);    \
  __macro(not_overflow, NotOverflowFunctor);
