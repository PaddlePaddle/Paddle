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
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MaskExtractCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* mask = ctx.Input<framework::LoDTensor>("Mask");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    auto* ids = ctx.Output<framework::LoDTensor>("Ids");
    auto* offset = ctx.Output<framework::LoDTensor>("Offset");

    int64_t out_len = 0;
    for (size_t i = 0; i < mask->dims()[0]; ++i) {
      if (mask->data<int64_t>()[i] >= 0) {
        out_len += 1;
      }
    }

    ids->Resize({out_len, 1});
    offset->Resize({out_len, 1});
    ids->mutable_data<int64_t>(ctx.GetPlace());
    offset->mutable_data<int64_t>(ctx.GetPlace());

    auto x_dims = x->dims();
    auto out_dims = x_dims;
    out_dims[0] = out_len;
    out->Resize(out_dims);
    out->mutable_data<T>(ctx.GetPlace());

    auto feat_num = x->numel() / x_dims[0];   
    int64_t out_idx = 0;
    platform::CPUPlace place = boost::get<platform::CPUPlace>(ctx.GetPlace()); 
    for (size_t i = 0; i < x_dims[0]; ++i) {
      if (mask->data<int64_t>()[i] >= 0) {
        offset->data<int64_t>()[out_idx] = i;
        ids->data<int64_t>()[out_idx] = mask->data<int64_t>()[i];
        memory::Copy(place, out->data<T>() + out_idx*feat_num, place,
                   x->data<T>()+i*feat_num, feat_num*sizeof(T));
        out_idx += 1;
      }
    }
  }
};


template <typename DeviceContext, typename T>
class MaskExtractCPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_out = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto* offset = ctx.Input<framework::LoDTensor>("Offset");
    auto* d_x = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));

    d_x->mutable_data<T>(ctx.GetPlace());
    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(ctx.template device_context<DeviceContext>(), d_x,
               static_cast<T>(0));
    auto x_dims = d_x->dims();
    auto feat_num = d_x->numel() / x_dims[0];   
    platform::CPUPlace place = boost::get<platform::CPUPlace>(ctx.GetPlace()); 
    for (int64_t i = 0; i < d_out->dims()[0]; ++i) {
        memory::Copy(place, d_x->data<T>() + offset->data<int64_t>()[i]*feat_num, place,
                   d_out->data<T>()+i*feat_num, feat_num*sizeof(T));
    }  
  }
};

}  // namespace operators
}  // namespace paddle
