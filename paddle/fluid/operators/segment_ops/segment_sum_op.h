/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/segment_ops/segment_pooling.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class SegmentSumKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<Tensor>("X");
    auto* segment = context.Input<Tensor>("SegmentIds");
    auto* out = context.Output<Tensor>("Out");
    // T pad_value = static_cast<T>(context.Attr<float>("pad_value"));

    auto dims = in->dims();
    Tensor* index = nullptr;
    std::string pooltype = "SUM";
    // SegmentSumFunctor<DeviceContext, T> pool;

    auto index_type = segment->type();
    if (index_type == framework::proto::VarType::INT32) {
      auto* segment_ids = segment->data<int>();
      dims[0] = static_cast<int64_t>(segment_ids[segment->numel() - 1] + 1);
      VLOG(4) << "the dims of result: " << dims;
      out->Resize({dims});

      out->mutable_data<T>(context.GetPlace());
      SegmentPoolFunctor<DeviceContext, T, int> pool;
      pool(context.template device_context<DeviceContext>(), *in, *segment, out,
           index, pooltype);
    } else if (index_type == framework::proto::VarType::INT64) {
      auto* segment_ids = segment->data<int64_t>();
      dims[0] = static_cast<int64_t>(segment_ids[segment->numel() - 1] + 1);
      VLOG(4) << "the dims of result: " << dims;
      out->Resize({dims});

      out->mutable_data<T>(context.GetPlace());
      SegmentPoolFunctor<DeviceContext, T, int64_t> pool;
      pool(context.template device_context<DeviceContext>(), *in, *segment, out,
           index, pooltype);
    } else {
      PADDLE_THROW("unsupported index type");
    }
  }
};

template <typename DeviceContext, typename T>
class SegmentSumGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* segment = context.Input<Tensor>("SegmentIds");
    auto* out_g = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* in_g = context.Output<Tensor>(framework::GradVarName("X"));
    std::string pooltype = "SUM";
    const Tensor* index = nullptr;
    in_g->mutable_data<T>(context.GetPlace());
    auto index_type = segment->type();
    if (index_type == framework::proto::VarType::INT32) {
      SegmentPoolGradFunctor<DeviceContext, T, int> pool;
      pool(context.template device_context<DeviceContext>(), *out_g, *segment,
           in_g, index, pooltype);
    } else if (index_type == framework::proto::VarType::INT64) {
      SegmentPoolGradFunctor<DeviceContext, T, int64_t> pool;
      pool(context.template device_context<DeviceContext>(), *out_g, *segment,
           in_g, index, pooltype);
    } else {
      PADDLE_THROW("unsupported index type");
    }
  }
};

}  // namespace operators
}  // namespace paddle
